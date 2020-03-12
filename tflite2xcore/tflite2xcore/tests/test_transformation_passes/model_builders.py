# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
import numpy
from copy import deepcopy
from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes, XCOREOpCodes


def build_elementwise_op(builtin_opcode, subgraph=None, *, input_shape, tensor_type):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    quantization = {'scale': [0.35], 'zero_point': [0]}
    tin = subgraph.create_tensor(
        'input', tensor_type, shape=input_shape, isinput=True,
        quantization=deepcopy(quantization))
    tout = subgraph.create_tensor(
        'output', tin.type, shape=tin.shape, isoutput=True,
        quantization=deepcopy(quantization))
    subgraph.create_operator(OperatorCode(builtin_opcode),
                             inputs=[tin], outputs=[tout])

    return subgraph.model


def build_relu(subgraph=None, **kwargs):
    return build_elementwise_op(BuiltinOpCodes.RELU, subgraph, **kwargs)


def build_relu6(subgraph=None, **kwargs):
    return build_elementwise_op(BuiltinOpCodes.RELU6, subgraph, **kwargs)


def build_tanh(subgraph=None, **kwargs):
    return build_elementwise_op(BuiltinOpCodes.TANH, subgraph, **kwargs)


def build_logistic(subgraph=None, **kwargs):
    return build_elementwise_op(BuiltinOpCodes.LOGISTIC, subgraph, **kwargs)


def build_abs(subgraph=None, **kwargs):
    return build_elementwise_op(BuiltinOpCodes.ABS, subgraph, **kwargs)


def build_mean(subgraph=None, *, input_shape, reduction_dims):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    tin = subgraph.create_tensor(
        'input', type_=TensorType.INT8, shape=input_shape, isinput=True)
    tred = subgraph.create_tensor(
        'reduction_dims', TensorType.INT32, [len(reduction_dims)])
    tout = subgraph.create_tensor(
        'output', tin.type, [tin.shape[0] + tin.shape[3]], isoutput=True)
    tred.buffer.data = numpy.array(reduction_dims, dtype=numpy.int32)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.MEAN),
                             inputs=[tin, tred], outputs=[tout])

    return subgraph.model


def build_argmax(subgraph=None, *, input_shape, input_type):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    tin = subgraph.create_tensor(
        'input', type_=input_type, shape=input_shape, isinput=True)
    tout = subgraph.create_tensor(
        'output', TensorType.INT32, tin.shape, isoutput=True)
    dim_tensor = subgraph.create_tensor(
        "axis", TensorType.INT32, shape=[])
    dim_tensor.buffer.data = numpy.int32([1])
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.ARG_MAX),
                             inputs=[tin, dim_tensor], outputs=[tout])

    return subgraph.model


def build_pool(builtin_opcode, subgraph=None, *, input_shape, padding, pool_size, strides):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    output_shape = [input_shape[0], input_shape[1] / 2, input_shape[1] / 2, input_shape[3]]
    quantization = {'scale': [0.35], 'zero_point': [0]}
    tin = subgraph.create_tensor(
        'input', TensorType.INT8, input_shape,
        isinput=True, quantization=deepcopy(quantization))
    tout = subgraph.create_tensor(
        'output', tin.type, output_shape,
        isoutput=True, quantization=deepcopy(quantization))

    op = subgraph.create_operator(
        OperatorCode(builtin_opcode), inputs=[tin], outputs=[tout])
    assert padding in ['SAME', 'VALID']
    assert len(strides) == len(pool_size) == 2
    op.builtin_options = {'padding': padding,
                          'stride_h': strides[0], 'stride_w': strides[1],
                          'filter_height': pool_size[0], 'filter_width': pool_size[1],
                          'fused_activation_function': 'NONE'}

    return subgraph.model


def build_maxpool(subgraph=None, **kwargs):
    return build_pool(BuiltinOpCodes.MAX_POOL_2D, subgraph, **kwargs)


def build_avgpool(subgraph=None, **kwargs):
    return build_pool(BuiltinOpCodes.AVERAGE_POOL_2D, subgraph, **kwargs)


def build_fc(subgraph=None, *, outputs, input_shape):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    weight_shape = [outputs, numpy.prod(input_shape[1:])]
    tin = subgraph.create_tensor(
        'input', TensorType.INT8, input_shape, isinput=True)
    w = subgraph.create_tensor(
        'weights', TensorType.INT8, weight_shape,
        quantization={'scale': [0.35], 'zero_point': [0]})
    b = subgraph.create_tensor(
        'biases', TensorType.INT32, weight_shape[:1])
    tout = subgraph.create_tensor(
        'output', tin.type, shape=[1, weight_shape[0]], isoutput=True)
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.FULLY_CONNECTED),
        inputs=[tin, w, b], outputs=[tout])

    return subgraph.model


def build_intermediate_fc(subgraph=None, *, outputs, input_shape):
    model = build_fc(subgraph, outputs=outputs, input_shape=input_shape)
    subgraph = subgraph or model.subgraphs[0]

    subgraph.get_tensor('weights').name = 'weights_1'
    subgraph.get_tensor('biases').name = 'biases_1'

    tmid = subgraph.get_tensor('output')
    tmid.name = 'intermediate'
    subgraph.outputs.remove(tmid)

    return model


def build_softmax(subgraph=None, *, outputs, input_shape):
    model = build_intermediate_fc(subgraph, outputs=outputs, input_shape=input_shape)
    subgraph = subgraph or model.subgraphs[0]
    tmid = subgraph.get_tensor('intermediate')

    tout = subgraph.create_tensor('output', tmid.type, tmid.shape, isoutput=True)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.SOFTMAX),
                             inputs=[tmid], outputs=[tout])

    return model


def build_mlp(subgraph=None, *, outputs, hidden_nodes, input_shape):
    model = build_intermediate_fc(subgraph, outputs=hidden_nodes, input_shape=input_shape)
    subgraph = subgraph or model.subgraphs[0]
    tmid = subgraph.get_tensor('intermediate')

    w2_shape = [outputs, hidden_nodes]
    w2 = subgraph.create_tensor('weights_2', TensorType.INT8, w2_shape,
                                quantization={'scale': [0.22], 'zero_point': [0]})
    b2 = subgraph.create_tensor('biases_2', TensorType.INT32, shape=[outputs])
    tout = subgraph.create_tensor('output', tmid.type, shape=[1, outputs], isoutput=True)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.FULLY_CONNECTED),
                             inputs=[tmid, w2, b2], outputs=[tout])

    return model


def build_conv2d(subgraph=None, *, weight_shape, input_size, padding, strides):
    subgraph = subgraph or XCOREModel().create_subgraph()

    height, width = input_size
    C_out, K_h, K_w, C_in = weight_shape

    input_shape = [1, height, width, C_in]
    tin = subgraph.create_tensor(
        'input', TensorType.INT8, input_shape, isinput=True)
    w = subgraph.create_tensor(
        'weights', TensorType.INT8, weight_shape)
    b = subgraph.create_tensor(
        'biases', TensorType.INT32, weight_shape[:1])

    if padding == 'SAME':
        output_shape = [1, height, width, C_out]
    elif padding == 'VALID':
        output_shape = [1,
                        int(numpy.ceil((height - K_h + 1) / strides[0])),
                        int(numpy.ceil((width - K_w + 1) / strides[1])),
                        C_out]

    tout = subgraph.create_tensor(
        'output', tin.type, shape=output_shape, isoutput=True)

    op = subgraph.create_operator(OperatorCode(BuiltinOpCodes.CONV_2D),
                                  inputs=[tin, w, b], outputs=[tout])
    op.builtin_options = {'padding': padding,
                          'stride_h': strides[0], 'stride_w': strides[1],
                          'dilation_w_factor': 1, 'dilation_h_factor': 1}

    return subgraph.model


def build_depthwise_conv2d(subgraph=None, *, weight_shape, input_size, padding, strides=(1, 1)):
    subgraph = subgraph or XCOREModel().create_subgraph()

    # NOTE: weight_shape uses channel order HWIM (following TensorFlow DepthwiseConv)
    height, width = input_size
    K_h, K_w, C_in, depth_multiplier = weight_shape
    C_out = C_in * depth_multiplier

    input_shape = [1, input_size[0], input_size[1], C_in]
    weight_shape = [1, K_h, K_w, C_out]
    tin = subgraph.create_tensor('input', TensorType.INT8, input_shape, isinput=True)
    w = subgraph.create_tensor('weights', TensorType.INT8, weight_shape)
    b = subgraph.create_tensor('biases', TensorType.INT32, shape=[C_out])

    if padding == 'SAME':
        output_shape = [1, height, width, C_out]
    elif padding == 'VALID':
        output_shape = [1,
                        int(numpy.ceil((height - K_h + 1) / strides[0])),
                        int(numpy.ceil((width - K_w + 1) / strides[1])),
                        C_out]
    tout = subgraph.create_tensor(
        'output', tin.type, output_shape, isoutput=True)

    op = subgraph.create_operator(OperatorCode(BuiltinOpCodes.DEPTHWISE_CONV_2D),
                                  inputs=[tin, w, b], outputs=[tout])
    op.builtin_options = {'padding': padding,
                          'depth_multiplier': depth_multiplier,
                          'stride_h': strides[0], 'stride_w': strides[1],
                          'dilation_w_factor': 1, 'dilation_h_factor': 1}

    return subgraph.model


def build_DIDO(subgraph=None, *, weight_shape, input_size, padding):
    subgraph = subgraph or XCOREModel().create_subgraph()

    height, width = input_size
    C_out, K_h, K_w, C_in = weight_shape

    input_shape = [1, height, width, C_in]
    tin = subgraph.create_tensor('input', TensorType.INT8, input_shape, isinput=True)
    w = subgraph.create_tensor('weights', TensorType.INT8,
                               shape=[C_out // 16, K_h, K_w, C_in // 32, 16, 32])
    b = subgraph.create_tensor('biases', TensorType.INT16, shape=[C_out // 16, 2, 16])
    s = subgraph.create_tensor('scales', TensorType.INT16, b.shape)

    if padding == 'SAME':
        output_shape = [1, height, width, C_out]
    elif padding == 'VALID':
        output_shape = [1, height - K_h + 1, width - K_w + 1, C_out]
    tout = subgraph.create_tensor('output', tin.type, output_shape, isoutput=True)

    op = subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_conv2d_deepin_deepout_relu),
        inputs=[tin, w, b, s], outputs=[tout])
    op.add_custom_options(padding=padding, stride_h=1, stride_w=1)

    return subgraph.model


def build_DW(subgraph=None, *, weight_shape, input_size, padding, strides):
    subgraph = subgraph or XCOREModel().create_subgraph()

    height, width = input_size
    K_h, K_w, C_in = weight_shape
    C_out = C_in

    input_shape = [1, height, width, C_in]
    bias_shape = [int(numpy.ceil(C_out / 16)), 5, 16]
    tin = subgraph.create_tensor('input', TensorType.INT8, input_shape, isinput=True)
    w = subgraph.create_tensor('weights', TensorType.INT8, weight_shape)
    b = subgraph.create_tensor('bss', TensorType.INT16, bias_shape)

    if padding == 'SAME':
        output_shape = [1, height, width, C_out]
    elif padding == 'VALID':
        output_shape = [1, height - K_h + 1, width - K_w + 1, C_out]
    tout = subgraph.create_tensor('output', tin.type, output_shape, isoutput=True)

    op = subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_conv2d_deepin_deepout_relu),
        inputs=[tin, w, b], outputs=[tout])
    op.add_custom_options(padding=padding, stride=[strides[0], strides[1]])

    return subgraph.model


def build_pad(subgraph=None, *, input_shape, paddings):
    assert len(paddings) == len(input_shape) == 4
    for j, p in enumerate(paddings):
        assert len(p) == 2, f"padding[{j}] is not a pair"

    subgraph = subgraph or XCOREModel().create_subgraph()

    output_shape = [i + sum(p) for i, p in zip(input_shape, paddings)]
    tin = subgraph.create_tensor(
        'input', TensorType.INT8, input_shape, isinput=True)
    tout = subgraph.create_tensor('output', tin.type, output_shape, isoutput=True)
    p = subgraph.create_tensor('paddings', TensorType.INT32, shape=[4, 2])
    p.buffer.data = numpy.int32(paddings)

    subgraph.create_operator(OperatorCode(BuiltinOpCodes.PAD),
                             inputs=[tin, p], outputs=[tout])

    return subgraph.model
