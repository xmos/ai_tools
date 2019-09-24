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


def generate_unique_tensor_name(subgraph, base_name, suffix):
    tensor_names = [t['name'] for t in subgraph['tensors']]

    j = 1
    tensor_name = base_name + suffix
    while tensor_name in tensor_names:
        tensor_name = base_name + "_" + str(j) + suffix
        j += 1

    return tensor_name
