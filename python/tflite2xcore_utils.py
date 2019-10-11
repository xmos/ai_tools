# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy as np
import struct


class XCOps():
    FC_DEEPIN_SHALLOWOUT_FINAL = 'XC_fc_deepin_shallowout_final'
    MAXPOOL2D_DEEP = 'XC_maxpool2d_deep'
    CONV2D_SHALLOWIN_DEEPOUT_RELU = 'XC_conv2d_shallowin_deepout_relu'
    CONV2D_DEEPIN_DEEPOUT_RELU = 'XC_conv2d_deepin_deepout_relu'
    ARGMAX_16 = 'XC_argmax_16'


def get_input_tensor(subgraph, op_ind, input_ind):
    t_ind = subgraph['operators'][op_ind]['inputs'][input_ind]
    return subgraph['tensors'][t_ind]


def get_buffer_data_of_tensor(model, tensor):
        buffer_ind = tensor['buffer']
        return model['buffers'][buffer_ind]['data']


def tensor_to_np(model, tensor):
    buffer_data = get_buffer_data_of_tensor(model, tensor)
    if tensor['type'] == 'INT8':
        arr = np.int32(np.int8(buffer_data))
    elif tensor['type'] == 'INT32':
        arr = np.int32([i[0] for i in struct.iter_unpack('i', bytearray(buffer_data))])
    else:
        raise NotImplementedError()
    return arr.reshape(tensor['shape'])


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


def is_opcode_in_model(model, opcode_str):
    for c in model['operator_codes']:
        if c['builtin_code'] == opcode_str:
            return True
        elif c['builtin_code'] == 'CUSTOM' and c['custom_code'] == opcode_str:
            return True
    return False


def find_referencing_ops(tensor_ind, operators, *,
                         as_inputs=True, as_outputs=True):
    ref_op_inds = set()
    for j, op in enumerate(operators):
        if ((as_inputs and tensor_ind in op['inputs'])
                or (as_outputs and tensor_ind in op['outputs'])):
            ref_op_inds.add(j)
    return ref_op_inds


def find_used_opcode_inds(model):
    return set(op['opcode_index']
               for subgraph in model['subgraphs']
               for op in subgraph['operators'])


def find_used_tensor_inds(subgraph):
    used_tensor_inds = set()
    for op in subgraph['operators']:
        used_tensor_inds.update(op['inputs'] + op['outputs'])
    return used_tensor_inds


def find_used_buffer_inds(model):
    tensor_buffers = set(tensor['buffer']
                         for subgraph in model['subgraphs']
                         for tensor in subgraph['tensors'])
    metadata_buffers = set(m['buffer'] for m in model['metadata'])
    return tensor_buffers | metadata_buffers


def clean_unused_opcodes(model):
    opcode_ind_map = {}
    new_opcodes = []

    # find used opcodes and build new opcode list
    used_opcode_inds = find_used_opcode_inds(model)
    while used_opcode_inds:
        opcode_ind = used_opcode_inds.pop()
        opcode_ind_map[opcode_ind] = len(new_opcodes)
        new_opcodes.append(model['operator_codes'][opcode_ind])
        
    # replace opcode list and update references
    model['operator_codes'] = new_opcodes
    for subgraph in model['subgraphs']:
        for op in subgraph['operators']:
            op['opcode_index'] = opcode_ind_map[op['opcode_index']]


def clean_unused_tensors(model):
    for subgraph in model['subgraphs']:
        tensor_ind_map = {}
        new_tensors = []

        # find used tensors and build new tensor list
        used_tensor_inds = find_used_tensor_inds(subgraph)
        while used_tensor_inds:
            tensor_ind = used_tensor_inds.pop()
            tensor_ind_map[tensor_ind] = len(new_tensors)
            new_tensors.append(subgraph['tensors'][tensor_ind])

        # replace tensor list and update references
        subgraph['tensors'] = new_tensors
        for op in subgraph['operators']:
            op['inputs'] = [tensor_ind_map[i] for i in op['inputs']]
            op['outputs'] = [tensor_ind_map[i] for i in op['outputs']]
        subgraph['inputs'] = [tensor_ind_map[i] for i in subgraph['inputs']]
        subgraph['outputs'] = [tensor_ind_map[i] for i in subgraph['outputs']]


def clean_unused_buffers(model):
    buffer_ind_map = {}
    new_buffers = []

    # find used buffers and build new buffer list
    used_buffer_inds = find_used_buffer_inds(model)
    while used_buffer_inds:
        buffer_ind = used_buffer_inds.pop()
        buffer_ind_map[buffer_ind] = len(new_buffers)
        new_buffers.append(model['buffers'][buffer_ind])

    # replace opcode list and update references
    model['buffers'] = new_buffers
    for metadata in model['metadata']:
        metadata['buffer'] = buffer_ind_map[metadata['buffer']]
    for subgraph in model['subgraphs']:
        for tensor in subgraph['tensors']:
            tensor['buffer'] = buffer_ind_map[tensor['buffer']]


def generate_unique_tensor_name(subgraph, base_name, suffix):
    tensor_names = [t['name'] for t in subgraph['tensors']]

    j = 1
    tensor_name = base_name + suffix
    while tensor_name in tensor_names:
        tensor_name = base_name + "_" + str(j) + suffix
        j += 1

    return tensor_name
