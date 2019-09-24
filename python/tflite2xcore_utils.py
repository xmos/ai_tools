# Copyright (c) 2019, XMOS Ltd, All rights reserved

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


def find_used_opcode_inds(model):
    return set(op['opcode_index']
               for subgraph in model['subgraphs']
               for op in subgraph['operators'])


def find_used_tensor_inds(model):
    used_tensor_inds = set()
    for subgraph in model['subgraphs']:
        for op in subgraph['operators']:
            used_tensor_inds.update(op['inputs'] + op['outputs'])
    return used_tensor_inds


def find_used_buffer_inds(model):
    return set(tensor['buffer']
               for subgraph in model['subgraphs']
               for tensor in subgraph['tensors'])


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
        used_tensor_inds = find_used_tensor_inds(model)
        while used_tensor_inds:
            tensor_ind = used_tensor_inds.pop()
            tensor_ind_map[tensor_ind] = len(new_tensors)
            new_tensors.append(subgraph['tensors'][tensor_ind])

        # replace tensor list and update references
        subgraph['tensors'] = new_tensors
        for op in subgraph['operators']:
            op['inputs'] = [tensor_ind_map[i] for i in op['inputs']]
            op['outputs'] = [tensor_ind_map[i] for i in op['outputs']]


def clean_unused_buffers(model):
    buffer_ind_map = {}
    new_buffers = []

    # find used buffers and build new buffer list
    used_buffer_inds = find_used_buffer_inds(model)
    while used_buffer_inds:
        buffer_ind = used_buffer_inds.pop()
        buffer_ind_map[buffer_ind] = len(new_buffers)
        new_buffers.append(model['buffers'][buffer_ind])

    # replace upcode list and update references
    model['buffers'] = new_buffers
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
