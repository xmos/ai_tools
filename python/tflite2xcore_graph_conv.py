#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

import json
import os
import shutil
import argparse
import tempfile

import numpy as np

from copy import deepcopy
from tflite2xcore_utils import DEFAULT_FLATC, DEFAULT_SCHEMA
from tflite2xcore_utils import load_tflite_as_json, save_json_as_tflite


def get_opcode_index(model, opcode_str):
    for j, opcode in enumerate(model['operator_codes']):
        if opcode['builtin_code'] == opcode_str:
            return j
    return None


def get_custom_opcode_index(model, opcode_str):
    for j, opcode in enumerate(model['operator_codes']):
        if 'custom_code' in opcode:
            if opcode['custom_code'] == opcode_str:
                return j
    return None


def find_referencing_ops(tensor_ind, operators, *,
                         as_inputs=True, as_outputs=True):
    ref_op_inds = set()

    for j, op in enumerate(operators):    
        if ((as_inputs and tensor_ind in op['inputs'])
            or (as_outputs and tensor_ind in op['outputs'])):
            ref_op_inds.add(j)

    return ref_op_inds
            

def get_replacements(model, mode=None):
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
        replacements, ops_to_remove = get_replacements(model, mode=mode)
        subgraph[mode] = [replacements[k] if k in replacements else ind
                          for k, ind in enumerate(subgraph[mode])]
        subgraph['operators'] = [op for j, op in enumerate(subgraph['operators'])
                                 if j not in ops_to_remove]


def replace_ops_with_XC(model):
    if len(model['subgraphs']) > 1:
        raise ValueError(
            "Number of subgraphs is {}, "
            "cannot be greater than 1.".format(len(model['subgraphs'])))

    subgraph = model['subgraphs'][0]
    tensors = subgraph['tensors']

    new_opcodes = set()
    op_replacement = {}

    for j, op in enumerate(subgraph['operators']):
        opcode_index = op['opcode_index']
        if opcode_index == get_opcode_index(model, 'FULLY_CONNECTED'):
            weight_tensor_ind = op['inputs'][1]  # the second tensor in the weight tensor
            buffer_ind = tensors[weight_tensor_ind]['buffer']
            tensor_shape = tensors[weight_tensor_ind]['shape']
            if tensor_shape[0] < 16 and tensor_shape[1] % 32 == 0:
                # shallow input, deep output fully connected layer
                custom_opcode = 'XC_fc_deepin_shallowout_lin'
                new_opcodes.add(custom_opcode)
                op_replacement[j] = custom_opcode                
                    
            else:
                print("WARNING: no replace rule for op FULLY_CONNECTED with shape {}".format(
                        tensor_shape))
        elif opcode_index == get_opcode_index(model, 'CONV_2D'):
            print('replace rule for CONV_2D not yet implemented')
        elif opcode_index == get_opcode_index(model, 'MAX_POOL_2D'):
            print('replace rule for MAX_POOL_2D not yet implemented')
        else:
            print("WARNING: no replace rule for op {}".format(
                model['operator_codes'][opcode_index]['builtin_code']))

    # add new opcodes
    while new_opcodes:
        model['operator_codes'].append(
            {'builtin_code': 32,  # 32 is code for CUSTOM, TODO: get this from original enum
             'custom_code': new_opcodes.pop(),
             'version': 1}
        )

    # replace operators
    for k, opcode_str in op_replacement.items():
        custom_opcode_ind = get_custom_opcode_index(model, opcode_str)
        subgraph['operators'][k]['opcode_index'] = custom_opcode_ind


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
    # TODO: clean unused opcodes
    # TODO: clean unused tensors
    # TODO: clean unused buffers    

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
