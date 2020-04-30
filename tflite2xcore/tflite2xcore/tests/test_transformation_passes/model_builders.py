# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
import numpy as np

from copy import deepcopy

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import (
    TensorType,
    OperatorCode,
    BuiltinOpCodes,
    XCOREOpCodes,
)


def build_split(subgraph=None, *, input_shape, tensor_type, axis, num_splits):
    assert 0 <= axis < len(input_shape)
    assert 1 < num_splits <= input_shape[axis]
    assert input_shape[axis] % num_splits == 0
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    tin = subgraph.create_tensor("input", tensor_type, shape=input_shape, isinput=True)
    t_axis = subgraph.create_tensor("axis", TensorType.INT32, shape=[])
    t_axis.buffer.data = np.array([axis], dtype=np.int32)

    out_shape = (
        *input_shape[:axis],
        int(input_shape[axis] // num_splits),
        *input_shape[axis + 1 :],
    )
    outputs = [
        subgraph.create_tensor(f"output_{j}", tin.type, out_shape, isoutput=True)
        for j in range(num_splits)
    ]
    op = subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.SPLIT), inputs=[t_axis, tin], outputs=outputs
    )
    op.builtin_options = {"num_splits": num_splits}

    return subgraph.model


def build_elementwise_op(builtin_opcode, subgraph=None, *, input_shape, tensor_type):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    quantization = {"scale": [0.35], "zero_point": [0]}
    tin = subgraph.create_tensor(
        "input",
        tensor_type,
        shape=input_shape,
        isinput=True,
        quantization=deepcopy(quantization),
    )
    tout = subgraph.create_tensor(
        "output",
        tin.type,
        shape=tin.shape,
        isoutput=True,
        quantization=deepcopy(quantization),
    )
    subgraph.create_operator(OperatorCode(builtin_opcode), inputs=[tin], outputs=[tout])

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
        "input", type_=TensorType.INT8, shape=input_shape, isinput=True
    )
    tred = subgraph.create_tensor(
        "reduction_dims", TensorType.INT32, [len(reduction_dims)]
    )
    tout = subgraph.create_tensor(
        "output", tin.type, [tin.shape[0] + tin.shape[3]], isoutput=True
    )
    tred.buffer.data = np.array(reduction_dims, dtype=np.int32)
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.MEAN), inputs=[tin, tred], outputs=[tout]
    )

    return subgraph.model


def build_argmax(subgraph=None, *, input_shape, input_type):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    tin = subgraph.create_tensor(
        "input", type_=input_type, shape=input_shape, isinput=True
    )
    tout = subgraph.create_tensor("output", TensorType.INT32, tin.shape, isoutput=True)
    dim_tensor = subgraph.create_tensor("axis", TensorType.INT32, shape=[])
    dim_tensor.buffer.data = np.int32([1])
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.ARG_MAX), inputs=[tin, dim_tensor], outputs=[tout]
    )

    return subgraph.model


def build_pool(
    builtin_opcode,
    subgraph=None,
    *,
    input_shape,
    padding,
    pool_size,
    strides,
    fused_activation="NONE",
):
    assert fused_activation in ["NONE", "RELU", "RELU6"]
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    output_shape = [
        input_shape[0],
        input_shape[1] / 2,
        input_shape[1] / 2,
        input_shape[3],
    ]
    quantization = {"scale": [0.35], "zero_point": [0]}
    tin = subgraph.create_tensor(
        "input",
        TensorType.INT8,
        input_shape,
        isinput=True,
        quantization=deepcopy(quantization),
    )
    tout = subgraph.create_tensor(
        "output",
        tin.type,
        output_shape,
        isoutput=True,
        quantization=deepcopy(quantization),
    )

    op = subgraph.create_operator(
        OperatorCode(builtin_opcode), inputs=[tin], outputs=[tout]
    )
    assert padding in ["SAME", "VALID"]
    assert len(strides) == len(pool_size) == 2
    op.builtin_options = {
        "padding": padding,
        "stride_h": strides[0],
        "stride_w": strides[1],
        "filter_height": pool_size[0],
        "filter_width": pool_size[1],
        "fused_activation_function": fused_activation,
    }

    return subgraph.model


def build_maxpool(subgraph=None, **kwargs):
    return build_pool(BuiltinOpCodes.MAX_POOL_2D, subgraph, **kwargs)


def build_avgpool(subgraph=None, **kwargs):
    return build_pool(BuiltinOpCodes.AVERAGE_POOL_2D, subgraph, **kwargs)


def build_fc(subgraph=None, *, outputs, input_shape):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    weight_shape = [outputs, np.prod(input_shape[1:])]

    tin = subgraph.create_tensor(
        "input",
        TensorType.INT8,
        input_shape,
        isinput=True,
        quantization={"scale": [0.02874], "zero_point": [-2]},
    )
    w = subgraph.create_tensor(
        "weights",
        TensorType.INT8,
        weight_shape,
        quantization={"scale": [0.00836], "zero_point": [0]},
    )
    b = subgraph.create_tensor(
        "biases",
        TensorType.INT32,
        weight_shape[:1],
        quantization={"scale": [0.00024], "zero_point": [0]},
    )
    tout = subgraph.create_tensor(
        "output",
        tin.type,
        shape=[1, weight_shape[0]],
        isoutput=True,
        quantization={"scale": [0.11332], "zero_point": [6]},
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.FULLY_CONNECTED), inputs=[tin, w, b], outputs=[tout]
    )

    # add dummy data so that the op can be mutated
    w.buffer.data = np.int8(np.arange(0, np.prod(w.shape)) % 255 - 127)
    b.buffer.data = np.arange(np.prod(b.shape), dtype=np.int32)

    return subgraph.model


def build_intermediate_fc(subgraph=None, *, outputs, input_shape):
    model = build_fc(subgraph, outputs=outputs, input_shape=input_shape)
    subgraph = subgraph or model.subgraphs[0]

    subgraph.get_tensor("weights").name = "weights_1"
    subgraph.get_tensor("biases").name = "biases_1"

    tmid = subgraph.get_tensor("output")
    tmid.name = "intermediate"
    subgraph.outputs.remove(tmid)

    return model


def build_softmax(subgraph=None, *, outputs, input_shape):
    model = build_intermediate_fc(subgraph, outputs=outputs, input_shape=input_shape)
    subgraph = subgraph or model.subgraphs[0]
    tmid = subgraph.get_tensor("intermediate")

    tout = subgraph.create_tensor("output", tmid.type, tmid.shape, isoutput=True)
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.SOFTMAX), inputs=[tmid], outputs=[tout]
    )

    return model


def build_mlp(subgraph=None, *, outputs, hidden_nodes, input_shape):
    model = build_intermediate_fc(
        subgraph, outputs=hidden_nodes, input_shape=input_shape
    )
    subgraph = subgraph or model.subgraphs[0]
    tmid = subgraph.get_tensor("intermediate")

    w2_shape = [outputs, hidden_nodes]
    w2 = subgraph.create_tensor(
        "weights_2",
        TensorType.INT8,
        w2_shape,
        quantization={"scale": [0.22], "zero_point": [0]},
    )
    b2 = subgraph.create_tensor("biases_2", TensorType.INT32, shape=[outputs])
    tout = subgraph.create_tensor(
        "output", tmid.type, shape=[1, outputs], isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.FULLY_CONNECTED),
        inputs=[tmid, w2, b2],
        outputs=[tout],
    )

    return model


def build_conv2d(subgraph=None, *, weight_shape, input_size, padding, strides):
    subgraph = subgraph or XCOREModel().create_subgraph()

    height, width = input_size
    C_out, K_h, K_w, C_in = weight_shape

    input_shape = [1, height, width, C_in]
    tin = subgraph.create_tensor("input", TensorType.INT8, input_shape, isinput=True)
    w = subgraph.create_tensor("weights", TensorType.INT8, weight_shape)
    b = subgraph.create_tensor("biases", TensorType.INT32, weight_shape[:1])

    # add dummy data so that the op can be mutated
    w.buffer.data = np.int8(np.arange(0, np.prod(w.shape)) % 255 - 127)
    b.buffer.data = np.arange(np.prod(b.shape), dtype=np.int32)

    if padding == "SAME":
        output_shape = [1, height, width, C_out]
    elif padding == "VALID":
        output_shape = [
            1,
            int(np.ceil((height - K_h + 1) / strides[0])),
            int(np.ceil((width - K_w + 1) / strides[1])),
            C_out,
        ]

    tout = subgraph.create_tensor("output", tin.type, shape=output_shape, isoutput=True)

    op = subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.CONV_2D), inputs=[tin, w, b], outputs=[tout]
    )
    op.builtin_options = {
        "padding": padding,
        "stride_h": strides[0],
        "stride_w": strides[1],
        "dilation_w_factor": 1,
        "dilation_h_factor": 1,
    }

    return subgraph.model


def build_depthwise_conv2d(
    subgraph=None, *, weight_shape, input_size, padding, strides=(1, 1)
):
    subgraph = subgraph or XCOREModel().create_subgraph()

    # NOTE: weight_shape uses channel order HWIM (following TensorFlow DepthwiseConv)
    height, width = input_size
    K_h, K_w, C_in, depth_multiplier = weight_shape
    C_out = C_in * depth_multiplier

    input_shape = [1, input_size[0], input_size[1], C_in]
    weight_shape = [1, K_h, K_w, C_out]
    tin = subgraph.create_tensor("input", TensorType.INT8, input_shape, isinput=True)
    w = subgraph.create_tensor("weights", TensorType.INT8, weight_shape)
    b = subgraph.create_tensor("biases", TensorType.INT32, shape=[C_out])

    # add dummy data so that the op can be mutated
    w.buffer.data = np.int8(np.arange(0, np.prod(w.shape)) % 255 - 127)
    b.buffer.data = np.arange(np.prod(b.shape), dtype=np.int32)

    if padding == "SAME":
        output_shape = [1, height, width, C_out]
    elif padding == "VALID":
        output_shape = [
            1,
            int(np.ceil((height - K_h + 1) / strides[0])),
            int(np.ceil((width - K_w + 1) / strides[1])),
            C_out,
        ]
    tout = subgraph.create_tensor("output", tin.type, output_shape, isoutput=True)

    op = subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.DEPTHWISE_CONV_2D),
        inputs=[tin, w, b],
        outputs=[tout],
    )
    op.builtin_options = {
        "padding": padding,
        "depth_multiplier": depth_multiplier,
        "stride_h": strides[0],
        "stride_w": strides[1],
        "dilation_w_factor": 1,
        "dilation_h_factor": 1,
    }

    return subgraph.model


def build_XC_conv2d_deep(subgraph=None, *, weight_shape, input_size, strides):
    subgraph = subgraph or XCOREModel().create_subgraph()

    height, width = input_size
    C_out, _, _, C_in = weight_shape

    input_shape = [1, height, width, C_in]
    bss_shape = [int(np.ceil(C_out / 16)), 5, 16]
    tin = subgraph.create_tensor("input", TensorType.INT8, input_shape, isinput=True)
    w = subgraph.create_tensor("weights", TensorType.INT8, weight_shape)
    b = subgraph.create_tensor("bss", TensorType.INT16, bss_shape)

    # valid padding
    pads = [
        (0, np.ceil((i - k) / s) * s - i + k)
        for s, i, k in zip(strides, input_size, weight_shape[1:3])
    ]
    out_size = [
        (i - k + p[0] + p[1]) / s + 1
        for p, s, i, k in zip(pads, strides, input_size, weight_shape[1:3])
    ]
    output_shape = [C_out, *out_size, C_in]
    tout = subgraph.create_tensor("output", tin.type, output_shape, isoutput=True)

    op = subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_conv2d_deep), inputs=[tin, w, b], outputs=[tout]
    )
    op.add_custom_options(
        pad=[pads[0][0], pads[1][0], -127], stride=[strides[0], strides[1]]
    )

    return subgraph.model


def build_XC_conv2d_depthwise(subgraph=None, *, weight_shape, input_size, strides):
    subgraph = subgraph or XCOREModel().create_subgraph()

    height, width = input_size
    C_in = weight_shape[2]

    input_shape = [1, height, width, C_in]
    bss_shape = [int(np.ceil(C_in / 16)), 5, 16]
    tin = subgraph.create_tensor("input", TensorType.INT8, input_shape, isinput=True)
    w = subgraph.create_tensor("weights", TensorType.INT8, weight_shape)
    b = subgraph.create_tensor("bss", TensorType.INT16, bss_shape)

    # valid padding
    pads = [
        (0, np.ceil((i - k) / s) * s - i + k)
        for s, i, k in zip(strides, input_size, weight_shape[:2])
    ]
    out_size = [
        (i - k + p[0] + p[1]) / s + 1
        for p, s, i, k in zip(pads, strides, input_size, weight_shape[:2])
    ]
    output_shape = [1, *out_size, C_in]
    tout = subgraph.create_tensor("output", tin.type, output_shape, isoutput=True)

    op = subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_conv2d_depthwise),
        inputs=[tin, w, b],
        outputs=[tout],
    )
    op.add_custom_options(
        pad=[pads[0][0], pads[1][0], -127], stride=[strides[0], strides[1]]
    )

    return subgraph.model


def build_pad(subgraph=None, *, input_shape, paddings):
    assert len(paddings) == len(input_shape) == 4
    for j, p in enumerate(paddings):
        assert len(p) == 2, f"padding[{j}] is not a pair"

    subgraph = subgraph or XCOREModel().create_subgraph()

    output_shape = [i + sum(p) for i, p in zip(input_shape, paddings)]
    tin = subgraph.create_tensor("unpadded", TensorType.INT8, input_shape, isinput=True)
    tout = subgraph.create_tensor("padded", tin.type, output_shape, isoutput=True)
    p = subgraph.create_tensor("paddings", TensorType.INT32, shape=[4, 2])
    p.buffer.data = np.int32(paddings)

    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.PAD), inputs=[tin, p], outputs=[tout]
    )

    return subgraph.model


def _glue_ops(op1, op2):
    subgraph = op1.subgraph
    assert subgraph is op2.subgraph

    old_input, old_output = op2.inputs[0], op1.outputs[0]
    op2.inputs[0] = old_output
    subgraph.remove_tensor(old_input)
    if old_output in subgraph.outputs:
        subgraph.outputs.remove(old_output)
    old_output.consumers.append(op2)


def build_consecutive_pads(subgraph=None, *, input_shape, paddings_1, paddings_2):
    model = build_pad(subgraph, input_shape=input_shape, paddings=paddings_1)
    subgraph = subgraph or model.subgraphs[0]

    build_pad(subgraph, input_shape=subgraph.outputs[0].shape, paddings=paddings_2)

    pad_1, pad_2 = subgraph.operators[:2]
    _glue_ops(pad_1, pad_2)

    return model


def build_padded_DW(subgraph=None, *, weight_shape, input_size, paddings, strides):
    input_shape = [1, *input_size, weight_shape[-1]]
    model = build_pad(subgraph, input_shape=input_shape, paddings=paddings)
    subgraph = subgraph or model.subgraphs[0]
    output_shape = subgraph.outputs[0].shape

    build_XC_conv2d_depthwise(
        subgraph,
        weight_shape=weight_shape,
        input_size=output_shape[1:3],
        strides=strides,
    )

    pad_op, conv_op = subgraph.operators[:2]
    _glue_ops(pad_op, conv_op)

    old_input = conv_op.inputs[0]
    pad_op.outputs[0].quantization = old_input.quantization
    pad_op.inputs[0].quantization = old_input.quantization

    return model
