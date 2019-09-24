#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

import json
import os
import argparse

from tflite_utils import DEFAULT_FLATC, DEFAULT_SCHEMA
from tflite_utils import load_tflite_as_json, save_json_as_tflite
from tflite2xcore_utils import get_opcode_index
from tflite2xcore_utils import find_referencing_ops
from tflite2xcore_utils import clean_unused_buffers, clean_unused_opcodes, clean_unused_tensors
from tflite2xcore_graph_replacers import replace_with_XC_fc_deepin_shallowout_lin
from tflite2xcore_graph_replacers import XCOps


def get_input_output_replacements(model, mode=None):
    modes = ['inputs', 'outputs']
    if mode not in modes:
        raise ValueError('mode must be one of {}'.format(modes))

    replacements = {}
    ops_to_remove = set()

    opcodes = model['operator_codes']
    subgraph = model['subgraphs'][0]
    tensors, operators = subgraph['tensors'], subgraph['operators']

    for k, tensor_ind in enumerate(subgraph[mode]):
        tensor = tensors[tensor_ind]
        if tensor['type'] == 'FLOAT32':
            # references as outputs are ignored when replacing inputs
            ref_op_inds = find_referencing_ops(tensor_ind, operators,
                                               as_outputs=(mode!='inputs'))

            if len(ref_op_inds) == 0:
                print(f"WARNING (while replacing float {mode}): "
                      f"ignoring float tensor {tensor_ind} "
                      "(not referenced by any operator).")
            elif len(ref_op_inds) > 1:
                print(f"WARNING (while replacing float {mode}): "
                      f"ignoring float tensor {tensor_ind} "
                      "(more than one referencing op).")
            else:
                op_ind = ref_op_inds.pop()
                op = operators[op_ind]
                opcode = opcodes[op['opcode_index']]['builtin_code']
                if mode == 'inputs' and opcode != 'QUANTIZE':
                    print(f"WARNING (while replacing float {mode}): "
                          f"ignoring float tensor {tensor_ind} "
                          f"(consumer is of type '{opcode}' != 'QUANTIZE').")
                if mode == 'outputs' and opcode != 'DEQUANTIZE':
                    print(f"WARNING (while replacing float {mode}): "
                          f"ignoring float tensor {tensor_ind} "
                          f"(source is of type '{opcode}' != 'DEQUANTIZE').")
                else:
                    ops_to_remove.add(op_ind)
                    replacements[k] = op['outputs' if mode == 'inputs' else 'inputs'][0]
        elif tensor['type'] != 'INT8':
            print(f"WARNING (while replacing float {mode}): "
                  f"ignoring tensor {tensor_ind} "
                  f"(has unsupported type '{tensor['type']}')")

    return replacements, ops_to_remove


def replace_float_inputs_outputs(model):
    subgraph = model['subgraphs'][0]
    for mode in ['inputs', 'outputs']:
        replacements, ops_to_remove = get_input_output_replacements(model, mode=mode)
        subgraph[mode] = [replacements[k] if k in replacements else ind
                          for k, ind in enumerate(subgraph[mode])]
        subgraph['operators'] = [op for j, op in enumerate(subgraph['operators'])
                                 if j not in ops_to_remove]


def get_ops_replacements(model, subgraph_ind):
    subgraph = model['subgraphs'][subgraph_ind]
    tensors = subgraph['tensors']

    new_opcodes = set()
    op_replacement = {}

    # TODO: refactor the logic here into mapper(s)?
    for j, op in enumerate(subgraph['operators']):
        opcode_index = op['opcode_index']

        if opcode_index == get_opcode_index(model, 'FULLY_CONNECTED'):
            weight_tensor_ind = op['inputs'][1]  # the second tensor in the weight tensor
            tensor_shape = tensors[weight_tensor_ind]['shape']

            if tensor_shape[0] < 16 and tensor_shape[1] % 32 == 0:
                # shallow input, deep output fully connected layer
                custom_opcode = XCOps.FC_DEEPIN_SHALLOWOUT_LIN
                new_opcodes.add(custom_opcode)
                op_replacement[j] = custom_opcode
            else:
                raise NotImplementedError(
                    f"No replace rule for FULLY_CONNECTED with shape {tensor_shape}")

        elif opcode_index == get_opcode_index(model, 'CONV_2D'):
            weight_tensor_ind = op['inputs'][1]  # the second tensor in the weight tensor
            tensor_shape = tensors[weight_tensor_ind]['shape']
            options = op['builtin_options']
            strides = (options['stride_h'], options['stride_w'])
            dilation = (options['dilation_h_factor'], options['dilation_w_factor'])

            if dilation != (1, 1):
                raise NotImplementedError(
                    f"No replace rule for CONV_2D with dilation {dilation}")
            elif strides != (1, 1):
                raise NotImplementedError(
                    f"No replace rule for CONV_2D with strides {strides}")
            elif options['padding'] != 'SAME':
                raise NotImplementedError(
                    f"No replace rule for CONV_2D with padding {options['padding']}")
            elif tensor_shape[1] % 2 == 0 or tensor_shape[2] % 2 == 0:
                raise NotImplementedError(
                    f"No replace rule for CONV_2D with (even) kernel shape {tensor_shape[1:3]}")
            elif tensor_shape[0] % 16 == 0 and tensor_shape[3] % 32 == 0:
                # TODO:
                print(f"WARNING: replace rule for '{XCOps.CONV2D_DEEPIN_DEEPOUT_RELU}' not yet implemented")
            elif tensor_shape[0] % 16 == 0 and tensor_shape[3] <= 4:
                if tensor_shape[2] > 8:
                    raise NotImplementedError(
                        "No replace rule for CONV_2D with deep output, "
                        f"shallow input {tensor_shape[3]} (<= 4), "
                        f"and kernel width {tensor_shape[2]} (> 8)")

                # TODO:
                print(f"WARNING: replace rule for '{XCOps.CONV2D_SHALLOWIN_DEEPOUT_RELU}' not yet implemented")
            else:
                print("WARNING: replace rule for op CONV_2D with shape {}".format(
                        tensor_shape))

        elif opcode_index == get_opcode_index(model, 'MAX_POOL_2D'):
            options = op['builtin_options']
            strides = (options['stride_h'], options['stride_w'])
            pool_size = (options['filter_height'], options['filter_width'])

            # TODO: maybe add sanity check for input/output tensor quantization matching?
            if options['padding'] != 'VALID':
                raise NotImplementedError(
                    f"No replace rule for MAX_POOL_2D with padding {options['padding']}")
            elif strides != (2, 2):
                raise NotImplementedError(
                    f"No replace rule for MAX_POOL_2D with strides {strides}")
            elif pool_size != (2, 2):
                raise NotImplementedError(
                    f"No replace rule for MAX_POOL_2D with pool size {pool_size}")
            elif options['fused_activation_function'] != 'NONE':
                raise NotImplementedError(
                    f"No replace rule for MAX_POOL_2D with fused activation {options['fused_activation_function']}")
            else:
                input_shape = tensors[op['inputs'][0]]['shape']
                if input_shape[3] % 32 == 0:
                    # TODO:
                    print(f"WARNING: replace rule for '{XCOps.MAXPOOL2D_DEEP}' not yet implemented")
                else:
                    raise NotImplementedError(
                        f"No replace rule for MAX_POOL_2D with {input_shape[3]} input channels")

        else:
            print("WARNING: no replace rule for op {}".format(
                model['operator_codes'][opcode_index]['builtin_code']))

    return op_replacement, new_opcodes


def replace_ops_with_XC(model):
    if len(model['subgraphs']) > 1:
        raise ValueError(
            "Number of subgraphs is {}, "
            "cannot be greater than 1.".format(len(model['subgraphs'])))

    op_replacement, new_opcodes = get_ops_replacements(model, subgraph_ind=0)

    # add new opcodes
    while new_opcodes:
        model['operator_codes'].append(
            {'builtin_code': 'CUSTOM',
             'custom_code': new_opcodes.pop(),
             'version': 1}
        )

    # replace operators
    for op_ind, opcode_str in op_replacement.items():
        if opcode_str == XCOps.FC_DEEPIN_SHALLOWOUT_LIN:
            replace_with_XC_fc_deepin_shallowout_lin(model, subgraph_ind=0, op_ind=op_ind)


def main(tflite_input, tflite_output, *,
         flatc_bin=DEFAULT_FLATC, schema=DEFAULT_SCHEMA):

    if not os.path.exists(schema):
        raise FileNotFoundError(
            "Sorry, schema file cannot be found at {}".format(schema))

    if flatc_bin is None:
        raise RuntimeError("Sorry, cannot find flatc")
    elif not os.path.exists(flatc_bin):
        raise RuntimeError(
            "Sorry, flatc is not available at {}".format(flatc_bin))

    model = load_tflite_as_json(tflite_input,
                                flatc_bin=flatc_bin, schema=schema)

    # run graph manipulations
    replace_float_inputs_outputs(model)
    replace_ops_with_XC(model)
    clean_unused_opcodes(model)
    clean_unused_tensors(model)
    clean_unused_buffers(model)

    save_json_as_tflite(model, tflite_output,
                        flatc_bin=flatc_bin, schema=schema)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_input', help='Input .tflite file.')
    parser.add_argument('tflite_output', help='Output .tflite file.')
    parser.add_argument('--flatc', required=False, default=None,
                        help='Path to flatc executable.')
    parser.add_argument('--schema', required=False, default=None,
                        help='Path to .fbs schema file.')
    args = parser.parse_args()

    tflite_input = os.path.realpath(args.tflite_input)
    tflite_output = os.path.realpath(args.tflite_output)
    flatc_bin = args.flatc if args.flatc else DEFAULT_FLATC
    schema = args.schema if args.schema else DEFAULT_SCHEMA

    main(tflite_input, tflite_output, flatc_bin=flatc_bin, schema=schema)
