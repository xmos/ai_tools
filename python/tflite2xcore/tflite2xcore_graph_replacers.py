# Copyright (c) 2019, XMOS Ltd, All rights reserved

import struct
import flatbuffers

import numpy as np
import tensorflow as tf

from copy import deepcopy

from tflite2xcore_utils import generate_unique_tensor_name, XCOps
from tflite2xcore_utils import get_custom_opcode_index, tensor_to_np
from tflite2xcore_utils import get_input_tensor, get_buffer_data_of_tensor


def replace_basic_op(model, subgraph_ind, op_ind, opcode_str):
    subgraph = model['subgraphs'][subgraph_ind]
    custom_opcode_ind = get_custom_opcode_index(model, opcode_str)
    op = subgraph['operators'][op_ind]
    op['opcode_index'] = custom_opcode_ind
    op['builtin_options_type'] = 'NONE'
    del op['builtin_options']


def replace_with_XC_maxpool2d_deep(model, subgraph_ind, op_ind):
    opcode_str = XCOps.MAXPOOL2D_DEEP
    replace_basic_op(model, subgraph_ind, op_ind, opcode_str)


def calculate_real_multiplier(output_tensor, bias_tensor):
    output_scale = output_tensor['quantization']['scale'][0]
    bias_scale = np.array(bias_tensor['quantization']['scale'])
    return bias_scale / output_scale


def calculate_unified_bias(weights, bias, input_zero_point, output_zero_point, multiplier):
    zero_point_bias = np.sum(weights * input_zero_point,
                             axis=tuple(j for j in range(1, len(weights.shape))))
    return bias - np.int32(zero_point_bias) + np.int32(np.round(output_zero_point / multiplier))


def calculate_shift_scale(multiplier, bias_size):
    # NOTE: VLMUL expects one factor in Q2.14
    # we have 1 <= scale < 2 represented in Q2.14
    rshift = -np.ceil(np.log2(multiplier)) + 1
    scale = np.round(2**14 * (multiplier * 2**rshift))

    for j in range(len(scale)):
        if scale[j] == 2**15:
            rshift[j] -= 1
            scale[j] /= 2
        # we are using 16 bits instead of 8 so we need to adjust the shift
        # NOTE: VDEPTH8 shifts down by 8 bits, not 7 as stated on some pages of the ISA
        rshift[j] -= 8

    if len(scale) == 1:
        rshift = np.repeat(rshift, bias_size)
        scale = np.repeat(scale, bias_size)
    return np.int16(rshift), np.int16(scale)


def add_XC_shift_scale(model, subgraph_ind, multiplier, op, opcode_str, bias_size):
    subgraph = model['subgraphs'][subgraph_ind]

    # quantize multiplier and get right shift/scale
    rshift, scale = calculate_shift_scale(multiplier, bias_size)

    if rshift.shape != scale.shape:
        raise ValueError(f"Shift and scale shapes don't match: {rshift.shape} != {scale.shape}")

    # add tensor and buffer for rshift
    op['inputs'].append(len(subgraph['tensors']))
    subgraph['tensors'].append({
        'shape': [2] + list(rshift.shape),
        'type': 'INT16',
        'buffer': len(model['buffers']),
        'name': generate_unique_tensor_name(
            subgraph, base_name=opcode_str, suffix='/shift_scale'),
        'is_variable': False
    })
    model['buffers'].append({
        'data': list(b''.join([struct.pack('h', a) for a in rshift]) +  # pylint: disable=not-an-iterable
                     b''.join([struct.pack('h', a) for a in scale]))  # pylint: disable=not-an-iterable
    })


def replace_with_XC_fc_deepin_shallowout_final(model, subgraph_ind, op_ind):
    opcode_str = XCOps.FC_DEEPIN_SHALLOWOUT_FINAL
    replace_basic_op(model, subgraph_ind, op_ind, opcode_str)

    subgraph = model['subgraphs'][subgraph_ind]
    op = subgraph['operators'][op_ind]

    # retrieve weights, and rename weight tensor
    weight_tensor = get_input_tensor(subgraph, op_ind, input_ind=1)
    weights = tensor_to_np(model, weight_tensor)

    # retrieve biases
    bias_tensor = get_input_tensor(subgraph, op_ind, input_ind=2)
    bias = tensor_to_np(model, bias_tensor)

    # retrieve input zero point
    input_tensor = get_input_tensor(subgraph, op_ind, input_ind=0)
    input_zero_point = np.int32(input_tensor['quantization']['zero_point'][0])

    # retreive output quantization
    output_tensor = subgraph['tensors'][op['outputs'][0]]
    output_zero_point = output_tensor['quantization']['zero_point'][0]

    # calculate real multiplier
    multiplier = calculate_real_multiplier(output_tensor, bias_tensor)

    # calculate and save a unified bias vector
    bias = calculate_unified_bias(weights, bias, input_zero_point, output_zero_point, multiplier)
    buffer_ind = bias_tensor['buffer']
    model['buffers'][buffer_ind]['data'] = list(bias.tostring())

    # rename bias tensor and change quantization mode to alert users to unusual layout
    bias_tensor['name'] = generate_unique_tensor_name(
        subgraph, base_name=opcode_str, suffix='/biases')
    bias_tensor['quantization']['details_type'] = 'CustomQuantization'

    # rename weight tensor
    # NOTE: no weight layout rearrangement is done for this op
    weight_tensor['name'] = generate_unique_tensor_name(
        subgraph, base_name=opcode_str, suffix='/weights')

    # rename output tensor, change type and quantization
    # NOTE: this is because the op is at the end of a network, and should be followed by argmax/softmax
    output_tensor['type'] = 'INT16'
    output_tensor['name'] = generate_unique_tensor_name(
        subgraph, base_name=opcode_str, suffix='/output')
    output_tensor['quantization'] = {
        'scale': [output_tensor['quantization']['scale'][0] / 2**8],
        'zero_point': [int(output_tensor['quantization']['zero_point'][0] * 2**8)],
        'details_type': "CustomQuantization",
        'quantized_dimension': 0
    }

    add_XC_shift_scale(model, subgraph_ind, multiplier, op, opcode_str, bias.size)


def rearrange_weight_quantization(weight_tensor, acc_period):
    def reshape_kernel_quantization(arr):
        arr = np.array(arr)
        arr = arr.reshape((arr.shape[0] // acc_period, acc_period))
        return np.flip(arr, axis=1).flatten().tolist()

    weight_quantization = weight_tensor['quantization']
    weight_quantization['scale'] = reshape_kernel_quantization(weight_quantization['scale'])
    weight_quantization['zero_point'] = reshape_kernel_quantization(weight_quantization['zero_point'])


def calculate_pad_biases(weights, unified_bias, input_zero_point):
    _, K_h, K_w, C_in = weights.shape
    assert K_h % 2 == 1
    assert K_w % 2 == 1
    pad_b = pad_t = K_h//2
    pad_l = pad_r = K_w//2

    K = tf.convert_to_tensor(weights, dtype=tf.float32)
    K = tf.transpose(K, perm=(1, 2, 3, 0))

    pad_template = tf.zeros(shape=(1, K_h, K_w, C_in), dtype=tf.float32)
    pad_template = tf.pad(pad_template,
                          paddings=[(0, 0), (pad_b, pad_t), (pad_l, pad_r), (0, 0)],
                          constant_values=input_zero_point)
    pad_biases = tf.nn.conv2d(input=pad_template,
                              filters=K,
                              strides=1,
                              padding='VALID')

    return tf.dtypes.cast(pad_biases + unified_bias.reshape((1, 1, -1)), dtype=tf.int32).numpy()


def replace_convolution_bias_tensor(model, subgraph, bias_tensor, opcode_str,
                                    weights, bias,
                                    input_zero_point, output_zero_point,
                                    multiplier):
    # calculate a unified bias vector and rearrange
    bias = calculate_unified_bias(weights, bias,
                                  input_zero_point, output_zero_point, multiplier)
    new_bias = calculate_pad_biases(weights, bias, input_zero_point)
    new_bias = np.uint8(list(new_bias.flatten().tostring())).reshape(list(new_bias.shape[1:])+[-1])
    new_bias = np.stack([new_bias[:, :, :, :2], new_bias[:, :, :, 2:]], axis=2)

    # save bias vector data, change type and shape
    buffer_ind = bias_tensor['buffer']
    bias_tensor['type'] = 'INT16'
    bias_tensor['shape'] = list(new_bias.shape[:-1])
    model['buffers'][buffer_ind]['data'] = list(new_bias.flatten().tostring())

    # rename bias tensor and change quantization mode to alert users to unusual layout
    bias_tensor['name'] = generate_unique_tensor_name(
        subgraph, base_name=opcode_str, suffix='/biases_padded')
    bias_tensor['quantization']['details_type'] = 'CustomQuantization'


def replace_with_XC_conv2d_deepin_deepout_relu(model, subgraph_ind, op_ind):
    opcode_str = XCOps.CONV2D_DEEPIN_DEEPOUT_RELU
    replace_basic_op(model, subgraph_ind, op_ind, opcode_str)

    subgraph = model['subgraphs'][subgraph_ind]
    op = subgraph['operators'][op_ind]

    # retrieve weights, and rename weight tensor
    weight_tensor = get_input_tensor(subgraph, op_ind, input_ind=1)
    weights = tensor_to_np(model, weight_tensor)

    # retrieve biases
    bias_tensor = get_input_tensor(subgraph, op_ind, input_ind=2)
    bias = tensor_to_np(model, bias_tensor)

    # retrieve input zero point
    input_tensor = get_input_tensor(subgraph, op_ind, input_ind=0)
    input_zero_point = np.int32(input_tensor['quantization']['zero_point'][0])

    # retreive output quantization
    output_tensor = subgraph['tensors'][op['outputs'][0]]
    output_zero_point = output_tensor['quantization']['zero_point'][0]

    # calculate real multiplier
    multiplier = calculate_real_multiplier(output_tensor, bias_tensor)

    replace_convolution_bias_tensor(model, subgraph, bias_tensor, opcode_str,
                                    weights, bias,
                                    input_zero_point, output_zero_point,
                                    multiplier)

    # rearrange weight tensor
    acc_period, ve = 16, 32
    new_shape = (weights.shape[0] // acc_period, acc_period,
                 weights.shape[1], weights.shape[2],
                 weights.shape[3] // ve, ve)
    weights = weights.reshape(new_shape)
    weights = np.transpose(np.flip(weights, axis=1), axes=(0, 2, 3, 4, 1, 5))
    new_shape = weights.shape
    weights = np.int8(weights.flatten())

    # save weight tensor and update shape
    buffer_ind = weight_tensor['buffer']
    model['buffers'][buffer_ind]['data'] = list(weights.tostring())
    weight_tensor['shape'] = list(new_shape)

    # rearrange weight quantization parameters
    rearrange_weight_quantization(weight_tensor, acc_period)

    # rename weight tensor
    weight_tensor['name'] = generate_unique_tensor_name(
        subgraph, base_name=opcode_str, suffix='/weights')

    add_XC_shift_scale(model, subgraph_ind, multiplier, op, opcode_str, bias.size)


def rearrange_depthwise_weights(model, subgraph_ind, op_ind):
    subgraph = model['subgraphs'][subgraph_ind]
    op = subgraph['operators'][op_ind]
    weight_tensor = subgraph['tensors'][op['inputs'][1]]
    weight_shape = weight_tensor['shape']

    buffer = model['buffers'][weight_tensor['buffer']]
    weights = np.uint8(buffer['data']).reshape(weight_shape)
    weights = np.transpose(weights, axes=(3, 1, 2, 0))
    buffer['data'] = weights.flatten().tolist()

    weight_tensor['shape'] = list(weights.shape)


def add_unpadded_shape(model, subgraph_ind, op, opcode_str, unpadded_shape):
    subgraph = model['subgraphs'][subgraph_ind]
    op['inputs'].append(len(subgraph['tensors']))
    subgraph['tensors'].append({
        'shape': [len(unpadded_shape)],
        'type': 'INT32',
        'buffer': len(model['buffers']),
        'name': generate_unique_tensor_name(
            subgraph, base_name=opcode_str, suffix='/unpadded_shape'),
        'is_variable': False
    })
    model['buffers'].append({
        'data': list(np.array(unpadded_shape, dtype=np.int32).tostring())
    })


def replace_with_XC_conv2d_shallowin_deepout_relu(model, subgraph_ind, op_ind,
                                                  *, from_depthwise=False):
    if from_depthwise:
        rearrange_depthwise_weights(model, subgraph_ind, op_ind)

    opcode_str = XCOps.CONV2D_SHALLOWIN_DEEPOUT_RELU
    replace_basic_op(model, subgraph_ind, op_ind, opcode_str)

    subgraph = model['subgraphs'][subgraph_ind]
    op = subgraph['operators'][op_ind]

    # retrieve weights, and rename weight tensor
    weight_tensor = get_input_tensor(subgraph, op_ind, input_ind=1)
    weights = tensor_to_np(model, weight_tensor)

    # retrieve biases
    bias_tensor = get_input_tensor(subgraph, op_ind, input_ind=2)
    bias = tensor_to_np(model, bias_tensor)

    # retrieve input zero point
    input_tensor = get_input_tensor(subgraph, op_ind, input_ind=0)
    input_zero_point = np.int32(input_tensor['quantization']['zero_point'][0])

    # retreive output quantization
    output_tensor = subgraph['tensors'][op['outputs'][0]]
    output_zero_point = output_tensor['quantization']['zero_point'][0]

    # calculate real multiplier
    multiplier = calculate_real_multiplier(output_tensor, bias_tensor)

    replace_convolution_bias_tensor(model, subgraph, bias_tensor, opcode_str,
                                    weights, bias,
                                    input_zero_point, output_zero_point,
                                    multiplier)

    # rearrange and zero pad weight tensor
    weights = np.pad(weights, pad_width=[(0, 0),
                                         (0, 0),
                                         (0, 8-weights.shape[2]),
                                         (0, 4-weights.shape[3])])
    acc_period = 16
    new_shape = (weights.shape[0] // acc_period, acc_period, weights.shape[1], 8, 4)
    weights = np.int8(weights.reshape(new_shape))
    weights = np.transpose(np.flip(weights, axis=1), axes=(0, 2, 1, 3, 4))

    # save weight tensor
    buffer_ind = weight_tensor['buffer']
    model['buffers'][buffer_ind]['data'] = list(weights.tostring())

    # rearrange weight quantization parameters
    rearrange_weight_quantization(weight_tensor, acc_period)

    # rename input tensor
    input_tensor['name'] = generate_unique_tensor_name(
        subgraph, base_name=opcode_str, suffix='/input')

    # rename weight tensor
    weight_tensor['name'] = generate_unique_tensor_name(
        subgraph, base_name=opcode_str, suffix='/weights')

    add_XC_shift_scale(model, subgraph_ind, multiplier, op, opcode_str, bias.size)

    # save tensor containing original shape
    # TODO: it would be better to store this as a custom option of the op
    add_unpadded_shape(model, subgraph_ind, op, opcode_str, weight_tensor['shape'])

    # update zero padded shapes
    weight_tensor['shape'] = list(weights.shape)
    input_tensor['shape'][3] = 4
