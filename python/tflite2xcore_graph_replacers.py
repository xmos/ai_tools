# Copyright (c) 2019, XMOS Ltd, All rights reserved

import struct

import numpy as np

from tflite2xcore_utils import generate_unique_tensor_name
from tflite2xcore_utils import get_custom_opcode_index


class XCOps():
    FC_DEEPIN_SHALLOWOUT_LIN = 'XC_fc_deepin_shallowout_lin'
    MAXPOOL2D_DEEP = 'XC_maxpool2d_deep'
    CONV2D_SHALLOWIN_DEEPOUT_RELU = 'XC_conv2d_shallowin_deepout_relu'
    CONV2D_DEEPIN_DEEPOUT_RELU = 'XC_conv2d_deepin_deepout_relu'


def replace_with_XC_maxpool2d_deep(model, subgraph_ind, op_ind):
    subgraph = model['subgraphs'][subgraph_ind]
    opcode_str = XCOps.MAXPOOL2D_DEEP

    custom_opcode_ind = get_custom_opcode_index(model, opcode_str)
    op = subgraph['operators'][op_ind]
    op['opcode_index'] = custom_opcode_ind
    op['builtin_options_type'] = 'NONE'
    del op['builtin_options']


def replace_with_XC_fc_deepin_shallowout_lin(model, subgraph_ind, op_ind):
    subgraph = model['subgraphs'][subgraph_ind]
    opcode_str = XCOps.FC_DEEPIN_SHALLOWOUT_LIN

    custom_opcode_ind = get_custom_opcode_index(model, opcode_str)
    op = subgraph['operators'][op_ind]
    op['opcode_index'] = custom_opcode_ind
    op['builtin_options_type'] = 'NONE'
    del op['builtin_options']

    def get_input_tensor(subgraph, op_ind, input_ind):
        t_ind = subgraph['operators'][op_ind]['inputs'][input_ind]
        return subgraph['tensors'][t_ind]

    def get_buffer_data_of_tensor(model, tensor):
        buffer_ind = tensor['buffer']
        return model['buffers'][buffer_ind]['data']

    # retrieve weights, and rename weight tensor
    weight_tensor = get_input_tensor(subgraph, op_ind, input_ind=1)
    buffer_data = get_buffer_data_of_tensor(model, weight_tensor)
    weights = np.int32(np.int8(buffer_data)).reshape(weight_tensor['shape'])
    weight_tensor['name'] = generate_unique_tensor_name(subgraph,
        base_name=opcode_str, suffix='/weights')

    # retrieve biases
    bias_tensor = get_input_tensor(subgraph, op_ind, input_ind=2)
    buffer_data = get_buffer_data_of_tensor(model, bias_tensor)
    bias = np.int32([i[0] for i in struct.iter_unpack('i', bytearray(buffer_data))])

    # retrieve input zero point
    input_tensor = get_input_tensor(subgraph, op_ind, input_ind=0)
    input_zero_point = input_tensor['quantization']['zero_point'][0]
    input_zero_point_vec = np.int32(input_zero_point * np.ones(weights.shape[1:]))

    # retreive output quantization
    t_ind = op['outputs'][0]
    output_tensor = subgraph['tensors'][t_ind]
    output_scale = output_tensor['quantization']['scale'][0]
    output_zero_point = output_tensor['quantization']['zero_point'][0]

    # calculate real multiplier
    bias_scale = np.array(bias_tensor['quantization']['scale'][0])  # TODO: this might be channelwise
    multiplier = bias_scale / output_scale

    # calculate and save a single bias vector
    new_bias = bias - np.matmul(weights, input_zero_point_vec) \
        + np.int32(output_zero_point / multiplier)
    buffer_ind = bias_tensor['buffer']
    model['buffers'][buffer_ind]['data'] = list(new_bias.tostring())
    bias_tensor['name'] = generate_unique_tensor_name(subgraph,
        base_name=opcode_str, suffix='/biases')

    # quantize multiplier to get right shift/scale
    # NOTE: VLMUL expects one factor in Q2.14
    rshift = -np.ceil(np.log2(multiplier))
    scale = np.round(2**14 * (multiplier * 2**rshift))
    if scale == 2**14:
        rshift -= 1
        scale /= 2
    rshift -= 7 # this is because we are using 15 bits instead of 8
    rshift = np.repeat(np.int16(rshift), new_bias.size)
    scale = np.repeat(np.int16(scale), new_bias.size)

    # add tensor and buffer for rshift
    op['inputs'].append(len(subgraph['tensors']))
    subgraph['tensors'].append({
        'shape': list(rshift.shape),
        'type': 'INT16',
        'buffer': len(model['buffers']),
        'name': generate_unique_tensor_name(subgraph, base_name=opcode_str, suffix='/rshift'),
        'is_variable': False
    })
    model['buffers'].append({
        'data': list(b''.join([struct.pack('h', a) for a in rshift]))
    })

    # add tensor and buffer for scale
    op['inputs'].append(len(subgraph['tensors']))
    subgraph['tensors'].append({
        'shape': list(scale.shape),
        'type': 'INT16',
        'buffer': len(model['buffers']),
        'name': generate_unique_tensor_name(subgraph, base_name=opcode_str, suffix='/scale'),
        'is_variable': False
    })
    model['buffers'].append({
        'data': list(b''.join([struct.pack('h', a) for a in scale]))
    })