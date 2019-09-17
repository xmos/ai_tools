#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

import json
import os
import shutil
import argparse
import numpy as np
from copy import deepcopy


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


def replace_ops(model, new_model):
    if len(model['subgraphs']) > 1:
        raise ValueError(
            "Number of subgraphs is {}, "
            "cannot be greater than 1.".format(len(model['subgraphs'])))

    subgraph = model['subgraphs'][0]
    new_subgraph = new_model['subgraphs'][0]
    tensors = subgraph['tensors']

    for j, op in enumerate(subgraph['operators']):
        opcode_index = op['opcode_index']
        if opcode_index == get_opcode_index(model, 'FULLY_CONNECTED'):
            weight_tensor_ind = op['inputs'][1]  # the second tensor if the weight tensor
            buffer_ind = tensors[weight_tensor_ind]['buffer']
            tensor_shape = tensors[weight_tensor_ind]['shape']
            if tensor_shape[0] < 16 and tensor_shape[1] % 32 == 0:
                # shallow input, deep output fully connected layer

                # nothing to do with the tensor if standard layout is used
                #weight_buffer_data = np.int8(model['buffers'][buffer_ind]['data'])
                #weight_buffer_data = weight_buffer_data.reshape(tensor_shape)
                #weight_buffer_data = np.flip(weight_buffer_data, axis=0)
                #new_model['buffers'][buffer_ind]['data'] = np.uint8(weight_buffer_data.flatten()).tolist()
            
                # now replace op with custom op
                custom_opcode = 'XC_fc_deepin_shallowout_lin'
                custom_opcode_ind = get_custom_opcode_index(model, custom_opcode)
                if custom_opcode_ind is None:
                    # add new custom opcode
                    custom_opcode_ind = len(new_model['operator_codes'])
                    new_model['operator_codes'].append(
                        {'builtin_code': 32,  # 32 is code for CUSTOM, TODO: get this from original enum
                         'custom_code': custom_opcode,
                         'version': 1}
                    )
                new_subgraph['operators'][j]['opcode_index'] = custom_opcode_ind
                    
            else:
                print("WARNING: no replace rule for op FULLY_CONNECTED with shape {}".format(
                        tensor_shape))
        elif opcode_index == get_opcode_index(model, 'CONV_2D'):
            print('replace rule for CONV_2D not yet implemented')
        elif opcode_index == get_opcode_index(model, 'MAX_POOL_2D'):
            print('replace rule for MAX_POOL_2D not yet implemented')
        else:
            print("WARNING: no replace rule for op {}".format(model['operator_codes'][opcode_index]['builtin_code']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_input', help='Input .tflite file.')
    parser.add_argument('tflite_output', help='Output .tflite file.')
    args = parser.parse_args()
    tflite_input = os.path.realpath(args.tflite_input)
    tflite_output = os.path.realpath(args.tflite_output)

    # TODO: refactor this and use in visualize.py
    schema = os.path.normpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'schema.fbs'
    ))
    if not os.path.exists(schema):
        raise FileNotFoundError(
            "Sorry, schema file cannot be found at {}".format(schema))

    flatc_bin = shutil.which("flatc")
    if flatc_bin is None:
        raise RuntimeError("Sorry, cannot find flatc")
    elif not os.path.exists(flatc_bin):
        raise RuntimeError(
            "Sorry, flatc is not available at {}".format(flatc_bin))

    # convert model to json, TODO: use tempfile for this dir instead of /tmp
    cmd = ("{flatc_bin} -t --strict-json --defaults-json "
           "-o /tmp {schema} -- {input}").format(
           flatc_bin=flatc_bin, input=tflite_input, schema=schema)
    print(cmd)
    os.system(cmd)

    out_file_base = os.path.join('/tmp', os.path.splitext(os.path.split(tflite_input)[-1])[0])

    # open json file
    json_input = out_file_base + ".json"
    with open(json_input, 'r') as f:
        model = json.load(f)
    new_model = deepcopy(model)

    # run graph manipulations
    replace_ops(model, new_model)
    #  TODO: remove unused op from opcode list

    # write new json file
    json_output = out_file_base + ".new.json"
    with open(json_output, 'w') as f:
        json.dump(new_model, f, indent=2)

    # convert to tflite
    cmd = ("{flatc_bin} -b --strict-json --defaults-json "
           "-o /tmp {schema} {json_output}").format(
           flatc_bin=flatc_bin, json_output=json_output, schema=schema)
    print(cmd)
    os.system(cmd)
    out_file_base = os.path.join('/tmp', os.path.splitext(os.path.split(tflite_input)[-1])[0])

    # move tflite output to where original file was
    tmp_tflite_output = out_file_base + ".new.tflite"
    shutil.move(tmp_tflite_output, tflite_output)
    
