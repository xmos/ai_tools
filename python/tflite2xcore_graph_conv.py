#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

import json
import os
import argparse

from tflite_utils import DEFAULT_FLATC, DEFAULT_SCHEMA
from tflite_utils import check_schema_path, check_flatc_path
from tflite_utils import load_tflite_as_json, save_json_as_tflite
from tflite2xcore_utils import get_opcode_index, find_referencing_ops
from tflite2xcore_utils import XCOps
from tflite2xcore_utils import clean_unused_buffers, clean_unused_opcodes, clean_unused_tensors
from tflite2xcore_utils import generate_unique_tensor_name

from tflite2xcore_graph_replacers import replace_with_XC_fc_deepin_shallowout_final
from tflite2xcore_graph_replacers import replace_with_XC_maxpool2d_deep
from tflite2xcore_graph_replacers import replace_with_XC_conv2d_deepin_deepout_relu
from tflite2xcore_graph_replacers import replace_with_XC_conv2d_shallowin_deepout_relu


def get_float_input_output_replacements(model, subgraph_ind, *, mode=None):
    modes = ['inputs', 'outputs']
    if mode not in modes:
        raise ValueError('mode must be one of {}'.format(modes))

    replacements = {}
    ops_to_remove = set()

    opcodes = model['operator_codes']
    subgraph = model['subgraphs'][subgraph_ind]
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


def remove_float_inputs_outputs(model):
    for subgraph_ind, subgraph in enumerate(model['subgraphs']):
        for mode in ['inputs', 'outputs']:
            replacements, ops_to_remove = get_float_input_output_replacements(
                model, subgraph_ind, mode=mode)
            subgraph[mode] = [replacements[k] if k in replacements else ind
                            for k, ind in enumerate(subgraph[mode])]
            subgraph['operators'] = [op for j, op in enumerate(subgraph['operators'])
                                    if j not in ops_to_remove]


def add_float_inputs_outputs(model):
    # add opcode indices if not already there
    quantize_opcode_ind = get_opcode_index(model, 'QUANTIZE')
    if quantize_opcode_ind is None:
        quantize_opcode_ind = len(model['operator_codes'])
        model['operator_codes'].append({
            'builtin_code': 'QUANTIZE',
            'version': 1
        })
    dequantize_opcode_ind = get_opcode_index(model, 'DEQUANTIZE')
    if dequantize_opcode_ind is None:
        dequantize_opcode_ind = len(model['operator_codes'])
        model['operator_codes'].append({
            'builtin_code': 'DEQUANTIZE',
            'version': 2
        })
    opcode_inds = {'inputs': quantize_opcode_ind, 'outputs': dequantize_opcode_ind}

    # TODO: this is a hack
    new_buffer_ind = 0
    model['buffers'].insert(new_buffer_ind, {})
    model['metadata'][0]['buffer'] = model['metadata'][0]['buffer'] + 1

    for subgraph in model['subgraphs']:
        for tensor in subgraph['tensors']:
            tensor['buffer'] = tensor['buffer'] + 1

        for mode in ['inputs', 'outputs']:
            for k, tensor_ind in enumerate(subgraph[mode]):
                tensor = subgraph['tensors'][tensor_ind]
                new_tensor_ind = len(subgraph['tensors'])
                if tensor['type'] in ['INT8', 'INT16', 'INT32']:
                    # add new op, new input/output tensor and buffer
                    subgraph['operators'].insert(
                        # not necessary to insert, could use append, but visualizer prefers this, so why not
                        0 if mode=='inputs' else len(subgraph['operators']),
                        {
                            'opcode_index': opcode_inds[mode],
                            mode: [new_tensor_ind],
                            'outputs' if mode=='inputs' else 'inputs': [tensor_ind],
                            'builtin_options_type': 'NONE',
                            'custom_options_format': 'FLEXBUFFERS'
                        }
                    )
                    subgraph['tensors'].append({
                        'shape': tensor['shape'],
                        'type': 'FLOAT32',
                        'buffer': new_buffer_ind,
                        'name': generate_unique_tensor_name(subgraph,
                            base_name=mode[:], suffix=''),
                        'is_variable': False
                    })
                    #model['buffers'].append({})

                    # update input/output list
                    subgraph[mode][k] = new_tensor_ind
                else:
                    print(f"WARNING (while adding float {mode}): "
                        f"ignoring tensor {tensor_ind} of type {tensor['type']} "
                        f"(supported types are ['INT8', 'INT16', 'INT32']).")


def remove_output_softmax(model):
    softmax_ind = get_opcode_index(model, 'SOFTMAX')
    for subgraph in model['subgraphs']:
        replacements = {}
        ops_to_remove = set()

        for op_ind, op in enumerate(subgraph['operators']):
            if op['opcode_index'] == softmax_ind:
                output_tensor_ind = op['outputs'][0]
                if output_tensor_ind in subgraph['outputs']:
                    ops_to_remove.add(op_ind)
                    replacements[output_tensor_ind] = op['inputs'][0]

        subgraph['outputs'] = [replacements[k] if k in replacements else ind
                               for k, ind in enumerate(subgraph['outputs'])]
        subgraph['operators'] = [op for j, op in enumerate(subgraph['operators'])
                                 if j not in ops_to_remove]


def add_output_argmax(model):
    opcode_str = XCOps.ARGMAX_16

    # add new opcode
    opcode_ind = len(model['operator_codes'])
    model['operator_codes'].append({
        'builtin_code': 'CUSTOM',
        'custom_code': opcode_str,
        'version': 1
    })

    for subgraph in model['subgraphs']:
        outputs = subgraph['outputs']
        if len(outputs) > 1:
            raise ValueError("Output argmax cannot be added to subgraphs with "
                             f"more than one output (found {len(outputs)})")

        # add new output tensor and buffer
        new_outputs = [len(subgraph['tensors'])]
        subgraph['tensors'].append({
            'shape': [1, 1],
            'type': 'INT32',
            'buffer': len(model['buffers']),
            'name': generate_unique_tensor_name(subgraph, base_name=opcode_str, suffix='/output'),
            'is_variable': False
        })
        model['buffers'].append({})
        subgraph['outputs'] = new_outputs
        
        # add argmax op to subgraph
        subgraph['operators'].append({
            'opcode_index': opcode_ind,
            'inputs': [outputs[0]],
            'outputs': new_outputs,
            'builtin_options_type': 'NONE',
            'custom_options_format': 'FLEXBUFFERS'
        })


def get_ops_replacements(model, subgraph_ind):
    subgraph = model['subgraphs'][subgraph_ind]
    tensors = subgraph['tensors']

    new_opcodes = set()
    op_replacement = []

    # TODO: refactor the logic here into mapper(s)?
    for j, op in enumerate(subgraph['operators']):
        opcode_index = op['opcode_index']

        if opcode_index == get_opcode_index(model, 'FULLY_CONNECTED'):
            weight_tensor_ind = op['inputs'][1]  # the second tensor in the weight tensor
            tensor_shape = tensors[weight_tensor_ind]['shape']

            if tensor_shape[0] < 16 and tensor_shape[1] % 32 == 0:
                # shallow input, deep output fully connected layer
                if op['outputs'][0] in subgraph['outputs']:
                    custom_opcode = XCOps.FC_DEEPIN_SHALLOWOUT_FINAL
                    new_opcodes.add(custom_opcode)
                    op_replacement.append({
                        "op_ind": j,
                        "old_opcode": 'FULLY_CONNECTED',
                        "new_opcode": custom_opcode
                    })
                else:
                    raise NotImplementedError(
                        f"No replace rule for FULLY_CONNECTED with shape {tensor_shape} as non-final layer")
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
                # deep input, deep output 2D convolution layer
                custom_opcode = XCOps.CONV2D_DEEPIN_DEEPOUT_RELU
                new_opcodes.add(custom_opcode)
                op_replacement.append({
                        "op_ind": j,
                        "old_opcode": 'CONV_2D',
                        "new_opcode": custom_opcode
                })
            elif tensor_shape[0] % 16 == 0 and tensor_shape[3] <= 4:
                if tensor_shape[2] > 8:
                    raise NotImplementedError(
                        "No replace rule for CONV_2D with deep output, "
                        f"shallow input {tensor_shape[3]} (<= 4), "
                        f"and kernel width {tensor_shape[2]} (> 8)")

                custom_opcode = XCOps.CONV2D_SHALLOWIN_DEEPOUT_RELU
                new_opcodes.add(custom_opcode)
                op_replacement.append({
                        "op_ind": j,
                        "old_opcode": 'CONV_2D',
                        "new_opcode": custom_opcode
                })
            else:
                raise NotImplementedError(
                    f"No replace rule for CONV_2D with shape {tensor_shape}")

        elif opcode_index == get_opcode_index(model, 'DEPTHWISE_CONV_2D'):
            weight_tensor_ind = op['inputs'][1]  # the second tensor in the weight tensor
            tensor_shape = tensors[weight_tensor_ind]['shape']
            options = op['builtin_options']
            strides = (options['stride_h'], options['stride_w'])
            dilation = (options['dilation_h_factor'], options['dilation_w_factor'])

            if dilation != (1, 1):
                raise NotImplementedError(
                    f"No replace rule for DEPTHWISE_CONV_2D with dilation {dilation}")
            elif strides != (1, 1):
                raise NotImplementedError(
                    f"No replace rule for DEPTHWISE_CONV_2D with strides {strides}")
            elif options['padding'] != 'SAME':
                raise NotImplementedError(
                    f"No replace rule for DEPTHWISE_CONV_2D with padding {options['padding']}")
            elif tensor_shape[1] % 2 == 0 or tensor_shape[2] % 2 == 0:
                raise NotImplementedError(
                    f"No replace rule for DEPTHWISE_CONV_2D with (even) kernel shape {tensor_shape[1:3]}")
            elif tensor_shape[0] > 1:
                raise NotImplementedError(
                    f"No replace rule for DEPTHWISE_CONV_2D for input channels {tensor_shape[0]} > 1")
            elif tensor_shape[3] % 16 == 0:
                if tensor_shape[2] > 8:
                    raise NotImplementedError(
                        "No replace rule for DEPTHWISE_CONV_2D with deep output, "
                        f"single input channel, and kernel width {tensor_shape[2]} (> 8)")
                
                custom_opcode = XCOps.CONV2D_SHALLOWIN_DEEPOUT_RELU
                new_opcodes.add(custom_opcode)
                op_replacement.append({
                        "op_ind": j,
                        "old_opcode": 'DEPTHWISE_CONV_2D',
                        "new_opcode": custom_opcode
                })
            else:
                raise NotImplementedError(
                    f"No replace rule for DEPTHWISE_CONV_2D with shape {tensor_shape}")

        elif opcode_index == get_opcode_index(model, 'MAX_POOL_2D'):
            options = op['builtin_options']
            strides = (options['stride_h'], options['stride_w'])
            pool_size = (options['filter_height'], options['filter_width'])
            input_tensor = tensors[op['inputs'][0]]
            output_tensor = tensors[op['outputs'][0]]

            if output_tensor['quantization'] != input_tensor['quantization']:
                raise ValueError("Input and output tensor quantization does not match for MAX_POOL_2D.")
            elif options['padding'] != 'VALID':
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
                input_shape = input_tensor['shape']
                if input_shape[3] % 32 == 0:
                    # deep maxpool2d layer
                    custom_opcode = XCOps.MAXPOOL2D_DEEP
                    new_opcodes.add(custom_opcode)
                    op_replacement.append({
                        "op_ind": j,
                        "old_opcode": 'MAX_POOL_2D',
                        "new_opcode": custom_opcode
                    })
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
    for replacement in op_replacement:
        if replacement['new_opcode'] == XCOps.FC_DEEPIN_SHALLOWOUT_FINAL:
            replace_with_XC_fc_deepin_shallowout_final(
                model, subgraph_ind=0, op_ind=replacement['op_ind'])
        elif replacement['new_opcode'] == XCOps.MAXPOOL2D_DEEP:
            replace_with_XC_maxpool2d_deep(
                model, subgraph_ind=0, op_ind=replacement['op_ind'])
        elif replacement['new_opcode'] == XCOps.CONV2D_DEEPIN_DEEPOUT_RELU:
            replace_with_XC_conv2d_deepin_deepout_relu(
                model, subgraph_ind=0, op_ind=replacement['op_ind'])
        elif replacement['new_opcode'] == XCOps.CONV2D_SHALLOWIN_DEEPOUT_RELU:
            replace_with_XC_conv2d_shallowin_deepout_relu(
                model, subgraph_ind=0, op_ind=replacement['op_ind'],
                from_depthwise=(replacement['old_opcode'] == 'DEPTHWISE_CONV_2D'))


def main(tflite_input, tflite_output, *,
         is_classifier=False, remove_softmax=False,
         flatc_bin=DEFAULT_FLATC, schema=DEFAULT_SCHEMA):

    check_schema_path(schema)
    check_flatc_path(flatc_bin)

    model = load_tflite_as_json(tflite_input,
                                flatc_bin=flatc_bin, schema=schema)

    # run graph manipulations
    remove_float_inputs_outputs(model)
    if remove_softmax or is_classifier:
        remove_output_softmax(model)

    replace_ops_with_XC(model)

    if is_classifier:
        add_output_argmax(model)

    clean_unused_opcodes(model)
    clean_unused_tensors(model)
    clean_unused_buffers(model)

    model['description'] = 'TOCO + XMOS converted.'

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
    parser.add_argument('--classifier',  action='store_true', default=False,
                        help="Apply optimizations for classifier networks "
                             "(e.g. softmax removal and output argmax).")
    parser.add_argument('--remove_softmax',  action='store_true', default=False,
                        help="Remove output softmax operation.")
    args = parser.parse_args()

    tflite_input = os.path.realpath(args.tflite_input)
    tflite_output = os.path.realpath(args.tflite_output)
    flatc_bin = args.flatc if args.flatc else DEFAULT_FLATC
    schema = args.schema if args.schema else DEFAULT_SCHEMA
    is_classifier = args.classifier
    remove_softmax = args.remove_softmax

    main(tflite_input, tflite_output,
         is_classifier=is_classifier, remove_softmax=remove_softmax,
         flatc_bin=flatc_bin, schema=schema)
