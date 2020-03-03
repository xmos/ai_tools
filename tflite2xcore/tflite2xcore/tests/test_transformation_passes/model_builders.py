# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
import numpy
from copy import deepcopy
from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes, XCOREOpCodes


def build_elementwise_op(builtin_opcode, *, input_shape, tensor_type):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    input_shape = [1] + list(input_shape)
    quantization = {'scale': [0.35], 'zero_point': [0]}
    tin = subgraph.create_tensor(
        'input', tensor_type, shape=input_shape, isinput=True,
        quantization=deepcopy(quantization))
    tout = subgraph.create_tensor(
        'output', tin.type, shape=tin.shape, isoutput=True,
        quantization=deepcopy(quantization))
    subgraph.create_operator(OperatorCode(builtin_opcode),
                             inputs=[tin], outputs=[tout])

    return model


def build_relu(**kwargs):
    return build_elementwise_op(BuiltinOpCodes.RELU, **kwargs)


def build_relu6(**kwargs):
    return build_elementwise_op(BuiltinOpCodes.RELU6, **kwargs)


def build_tanh(**kwargs):
    return build_elementwise_op(BuiltinOpCodes.TANH, **kwargs)


def build_logistic(**kwargs):
    return build_elementwise_op(BuiltinOpCodes.LOGISTIC, **kwargs)


def build_abs(**kwargs):
    return build_elementwise_op(BuiltinOpCodes.ABS, **kwargs)


def build_mean(*, input_shape, reduction_dims):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    input_shape = [1] + list(input_shape)
    tin = subgraph.create_tensor(
        'input', type_=TensorType.INT8, shape=input_shape, isinput=True)
    tred = subgraph.create_tensor(
        'reduction_dims', TensorType.INT32, [len(reduction_dims)])
    tout = subgraph.create_tensor(
        'output', tin.type, [tin.shape[0] + tin.shape[3]], isoutput=True)
    tred.buffer.data = numpy.array(reduction_dims, dtype=numpy.int32)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.MEAN),
                             inputs=[tin, tred], outputs=[tout])

    return model


def build_argmax(*, input_shape, input_type):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    input_shape = [1] + list(input_shape)
    tin = subgraph.create_tensor(
        'input', type_=input_type, shape=input_shape, isinput=True)
    tout = subgraph.create_tensor(
        'output', TensorType.INT32, tin.shape, isoutput=True)
    dim_tensor = subgraph.create_tensor(
        "axis", TensorType.INT32, shape=[])
    dim_tensor.buffer.data = numpy.int32([1])
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.ARG_MAX),
                             inputs=[tin, dim_tensor], outputs=[tout])

    return model


def build_pool(builtin_opcode, *, input_shape, padding, pool_size, strides):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    input_shape = [1] + list(input_shape)
    output_shape = [input_shape[0], input_shape[1] / 2, input_shape[1] / 2, input_shape[3]]
    quantization = {'scale': [0.35], 'zero_point': [0]}
    tin = subgraph.create_tensor(
        'input', TensorType.INT8,
        shape=input_shape, isinput=True, quantization=deepcopy(quantization))
    tout = subgraph.create_tensor(
        'output', tin.type,
        shape=output_shape, isoutput=True, quantization=deepcopy(quantization))

    op = subgraph.create_operator(
        OperatorCode(builtin_opcode), inputs=[tin], outputs=[tout])
    assert padding in ['SAME', 'VALID']
    assert len(strides) == len(pool_size) == 2
    op.builtin_options = {'padding': padding,
                          'stride_h': strides[0], 'stride_w': strides[1],
                          'filter_height': pool_size[0], 'filter_width': pool_size[1],
                          'fused_activation_function': 'NONE'}

    return model


def build_maxpool(**kwargs):
    return build_pool(BuiltinOpCodes.MAX_POOL_2D, **kwargs)


def build_avgpool(**kwargs):
    return build_pool(BuiltinOpCodes.AVERAGE_POOL_2D, **kwargs)


def build_fc(*, outputs, input_size):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    input_shape = [1] + list(input_size)
    weight_shape = [outputs, numpy.prod(input_shape[1:])]
    tin = subgraph.create_tensor(
        'input', TensorType.INT8, shape=input_shape, isinput=True)
    w = subgraph.create_tensor(
        'weights', TensorType.INT8, shape=weight_shape,
        quantization={'scale': [0.35], 'zero_point': [0]})
    b = subgraph.create_tensor(
        'biases', TensorType.INT32, shape=weight_shape[:1])
    tout = subgraph.create_tensor(
        'output', tin.type, shape=[1, weight_shape[0]], isoutput=True)
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.FULLY_CONNECTED),
        inputs=[tin, w, b], outputs=[tout])

    return model


def build_intermediate_fc(*, outputs, input_size):
    model = build_fc(outputs=outputs, input_size=input_size)
    subgraph = model.subgraphs[0]

    subgraph.get_tensor('weights').name = 'weights_1'
    subgraph.get_tensor('biases').name = 'biases_1'

    tmid = subgraph.get_tensor('output')
    tmid.name = 'intermediate'
    subgraph.outputs.remove(tmid)

    return model


def build_softmax(*, outputs, input_size):
    model = build_intermediate_fc(outputs=outputs, input_size=input_size)
    subgraph = model.subgraphs[0]
    tmid = subgraph.get_tensor('intermediate')

    tout = subgraph.create_tensor('output', tmid.type, shape=tmid.shape, isoutput=True)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.SOFTMAX),
                             inputs=[tmid], outputs=[tout])

    return model


def build_mlp(*, outputs, hidden_nodes, input_size):
    model = build_intermediate_fc(outputs=hidden_nodes, input_size=input_size)
    subgraph = model.subgraphs[0]
    tmid = subgraph.get_tensor('intermediate')

    w2_shape = [outputs, hidden_nodes]
    w2 = subgraph.create_tensor('weights_2', TensorType.INT8, shape=w2_shape,
                                quantization={'scale': [0.22], 'zero_point': [0]})
    b2 = subgraph.create_tensor('biases_2', TensorType.INT32, shape=[outputs])
    tout = subgraph.create_tensor('output', tmid.type, shape=[1, outputs], isoutput=True)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.FULLY_CONNECTED),
                             inputs=[tmid, w2, b2], outputs=[tout])

    return model


def build_conv2d(*, weight_shape, input_size, padding, strides):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    height, width = input_size
    C_out, K_h, K_w, C_in = weight_shape

    input_shape = [1, height, width, C_in]
    tin = subgraph.create_tensor(
        'input', TensorType.INT8, shape=input_shape, isinput=True)
    w = subgraph.create_tensor(
        'weights', TensorType.INT8, shape=weight_shape)
    b = subgraph.create_tensor(
        'biases', TensorType.INT32, shape=weight_shape[:1])

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

    return model


def build_depthwise_conv2d(*, weight_shape, input_size, padding, strides=(1, 1)):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    input_shape = [1, input_size[0], input_size[1], weight_shape[-1]]
    tin = subgraph.create_tensor('input', TensorType.INT8, shape=input_shape, isinput=True)
    w = subgraph.create_tensor('weights', TensorType.INT8, shape=weight_shape)
    b = subgraph.create_tensor('biases', TensorType.INT32, shape=weight_shape[:1])
    tout = subgraph.create_tensor(
        'output', tin.type, shape=input_shape[:-1] + weight_shape[:1], isoutput=True)

    op = subgraph.create_operator(OperatorCode(BuiltinOpCodes.DEPTHWISE_CONV_2D),
                                  inputs=[tin, w, b], outputs=[tout])
    op.builtin_options = {'padding': padding,
                          'stride_h': strides[0], 'stride_w': strides[1],
                          'dilation_w_factor': 1, 'dilation_h_factor': 1}

    return model


def build_DIDO(*, weight_shape, input_size, padding):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    height, width = input_size
    C_out, K_h, K_w, C_in = weight_shape

    input_shape = [1, height, width, C_in]
    tin = subgraph.create_tensor('input', TensorType.INT8, shape=input_shape, isinput=True)
    w = subgraph.create_tensor('weights', TensorType.INT8,
                               shape=[C_out // 16, K_h, K_w, C_in // 32, 16, 32])
    b = subgraph.create_tensor('biases', TensorType.INT16, shape=[C_out // 16, 2, 16])
    s = subgraph.create_tensor('scales', TensorType.INT16, shape=deepcopy(b.shape))

    if padding == 'SAME':
        output_shape = [1, height, width, C_out]
    elif padding == 'VALID':
        output_shape = [1, height - K_h + 1, width - K_w + 1, C_out]
    tout = subgraph.create_tensor('output', TensorType.INT8, shape=output_shape)

    op = subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_conv2d_deepin_deepout_relu),
        inputs=[tin, w, b, s], outputs=[tout])
    op.add_custom_options(padding=padding, stride_h=1, stride_w=1)

    return model
