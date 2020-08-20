# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
import numpy as np

from copy import deepcopy

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import (
    ActivationFunctionType,
    Padding,
    TensorType,
    OperatorCode,
    BuiltinOpCodes,
    XCOREOpCodes,
    BuiltinOptions,
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


def build_dequantize(subgraph=None, *, input_shape):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    qin = subgraph.create_tensor("input", TensorType.INT8, input_shape, isinput=True)
    fout = subgraph.create_tensor(
        "output_dequantized", TensorType.FLOAT32, qin.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qin], outputs=[fout]
    )

    return subgraph.model


def build_quantize(subgraph=None, *, input_shape):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    fin = subgraph.create_tensor("input", TensorType.FLOAT32, input_shape, isinput=True)
    qout = subgraph.create_tensor(
        "output_quantized", TensorType.INT8, fin.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin], outputs=[qout]
    )

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


def build_XC_avgpool2d_global(subgraph=None, *, input_shape, reduction_dims):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    tin = subgraph.create_tensor(
        "input", type_=TensorType.INT8, shape=input_shape, isinput=True
    )
    tred = subgraph.create_tensor(
        "reduction_dims", TensorType.INT32, [len(reduction_dims)]
    )
    tout = subgraph.create_tensor(
        "output", tin.type, [tin.shape[0], tin.shape[3]], isoutput=True
    )
    tred.buffer.data = np.array(reduction_dims, dtype=np.int32)
    subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_avgpool2d_global),
        inputs=[tin, tred],
        outputs=[tout],
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
    fused_activation=ActivationFunctionType.NONE,
):
    assert len(strides) == len(pool_size) == 2
    assert padding in Padding
    assert fused_activation in [
        ActivationFunctionType.NONE,
        ActivationFunctionType.RELU,
        ActivationFunctionType.RELU6,
    ]
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


def build_XC_pool(opcode, subgraph=None, *, input_shape, pool_size, strides):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, *input_shape]
    output_shape = [
        input_shape[0],
        input_shape[1] // 2,
        input_shape[1] // 2,
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

    op = subgraph.create_operator(OperatorCode(opcode), inputs=[tin], outputs=[tout])
    op.add_custom_options(
        pool=[pool_size[0], pool_size[0]], stride=[strides[0], strides[1]]
    )

    return subgraph.model


def build_XC_maxpool2d(subgraph=None, **kwargs):
    return build_XC_pool(XCOREOpCodes.XC_maxpool2d, subgraph, **kwargs)


def build_XC_avgpool2d(subgraph=None, **kwargs):
    return build_XC_pool(XCOREOpCodes.XC_avgpool2d, subgraph, **kwargs)


def build_fc(subgraph=None, *, outputs, input_shape, add_batch_dim=True):
    subgraph = subgraph or XCOREModel().create_subgraph()

    if add_batch_dim:
        # TODO unify this behaviour
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


def build_XC_fc_deepin_anyout(subgraph=None, *, outputs, input_channels):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, input_channels, 1, 1]
    weight_shape = [outputs, np.prod(input_shape[1:])]
    bso_shape = [int(np.ceil(outputs / 16)), 7, 16]

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
        bso_shape,
        quantization={"scale": [0.00024], "zero_point": [0]},
    )
    tout = subgraph.create_tensor(
        "output",
        TensorType.INT16,
        shape=[1, weight_shape[0]],
        isoutput=True,
        quantization={"scale": [0.11332], "zero_point": [6]},
    )
    subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_fc_deepin_anyout),
        inputs=[tin, w, b],
        outputs=[tout],
    )

    return subgraph.model


def build_XC_requantize_16_to_8(subgraph=None, *, outputs, input_channels):
    subgraph = subgraph or XCOREModel().create_subgraph()

    input_shape = [1, input_channels, 1, 1]
    weight_shape = [outputs, np.prod(input_shape[1:])]

    tin = subgraph.create_tensor(
        "input",
        TensorType.INT8,
        input_shape,
        isinput=True,
        quantization={"scale": [0.02874], "zero_point": [-2]},
    )
    tout = subgraph.create_tensor(
        "output",
        TensorType.INT16,
        shape=[1, weight_shape[0]],
        isoutput=True,
        quantization={"scale": [0.11332], "zero_point": [6]},
    )
    subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_requantize_16_to_8), inputs=[tin], outputs=[tout]
    )

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


def build_conv2d(subgraph=None, *, weight_shape, input_size, padding, strides):
    subgraph = subgraph or XCOREModel().create_subgraph()
    assert padding in Padding

    height, width = input_size
    C_out, K_h, K_w, C_in = weight_shape

    input_shape = [1, height, width, C_in]
    tin = subgraph.create_tensor(
        "input",
        TensorType.INT8,
        input_shape,
        isinput=True,
        quantization={"scale": [0.63], "zero_point": [-5]},
    )
    np.random.seed(42)
    w = subgraph.create_tensor(
        "weights",
        TensorType.INT8,
        weight_shape,
        quantization={
            "scale": np.random.uniform(size=(C_out,)).astype(float).tolist(),
            "zero_point": [0] * C_out,
        },
    )
    b = subgraph.create_tensor(
        "biases",
        TensorType.INT32,
        shape=[C_out],
        quantization={
            "scale": [
                tin.quantization["scale"][0] * s for s in w.quantization["scale"]
            ],
            "zero_point": [0] * C_out,
        },
    )

    # add dummy data so that the op can be mutated
    w.buffer.data = np.int8(np.arange(0, np.prod(w.shape)) % 255 - 127)
    b.buffer.data = np.arange(np.prod(b.shape), dtype=np.int32)

    if padding is Padding.SAME:
        output_shape = [1, height, width, C_out]
    elif padding is Padding.VALID:
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
    assert len(strides) == 2
    assert padding in Padding
    subgraph = subgraph or XCOREModel().create_subgraph()

    # NOTE: weight_shape uses channel order HWIM (following TensorFlow DepthwiseConv)
    height, width = input_size
    K_h, K_w, C_in, depth_multiplier = weight_shape
    C_out = C_in * depth_multiplier

    input_shape = [1, input_size[0], input_size[1], C_in]
    weight_shape = [1, K_h, K_w, C_out]
    tin = subgraph.create_tensor(
        "input",
        TensorType.INT8,
        input_shape,
        isinput=True,
        quantization={"scale": [0.48], "zero_point": [15]},
    )
    np.random.seed(42)
    w = subgraph.create_tensor(
        "weights",
        TensorType.INT8,
        weight_shape,
        quantization={
            "scale": np.random.uniform(size=(C_out,)).astype(float).tolist(),
            "zero_point": [0] * C_out,
        },
    )
    b = subgraph.create_tensor(
        "biases",
        TensorType.INT32,
        shape=[C_out],
        quantization={
            "scale": [
                tin.quantization["scale"][0] * s for s in w.quantization["scale"]
            ],
            "zero_point": [0] * C_out,
        },
    )

    # add dummy data so that the op can be mutated
    w.buffer.data = np.int8(np.arange(0, np.prod(w.shape)) % 255 - 127)
    b.buffer.data = np.arange(np.prod(b.shape), dtype=np.int32)

    if padding is Padding.SAME:
        output_shape = [1, height, width, C_out]
    elif padding is Padding.VALID:
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


def build_XC_conv2d(opcode, subgraph=None, *, weight_shape, input_size, strides):
    subgraph = subgraph or XCOREModel().create_subgraph()

    height, width = input_size
    C_out, _, _, C_in = weight_shape

    input_shape = [1, height, width, C_in]
    bso_shape = [int(np.ceil(C_out / 16)), 7, 16]
    tin = subgraph.create_tensor("input", TensorType.INT8, input_shape, isinput=True)
    w = subgraph.create_tensor("weights", TensorType.INT8, weight_shape)
    b = subgraph.create_tensor("bso", TensorType.INT16, bso_shape)

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
        OperatorCode(opcode), inputs=[tin, w, b], outputs=[tout]
    )
    op.add_custom_options(
        pad=[pads[0][0], pads[1][0], -127], stride=[strides[0], strides[1]]
    )

    return subgraph.model


def build_XC_conv2d_deep(subgraph=None, **kwargs):
    return build_XC_conv2d(XCOREOpCodes.XC_conv2d_deep, subgraph, **kwargs)


def build_XC_conv2d_shallowin(subgraph=None, **kwargs):
    return build_XC_conv2d(XCOREOpCodes.XC_conv2d_shallowin, subgraph, **kwargs)


def build_XC_conv2d_1x1(subgraph=None, **kwargs):
    return build_XC_conv2d(XCOREOpCodes.XC_conv2d_1x1, subgraph, **kwargs)


def build_XC_conv2d_depthwise(subgraph=None, *, weight_shape, input_size, strides):
    subgraph = subgraph or XCOREModel().create_subgraph()

    height, width = input_size
    C_in = weight_shape[2]

    input_shape = [1, height, width, C_in]
    bso_shape = [int(np.ceil(C_in / 16)), 7, 16]
    tin = subgraph.create_tensor("input", TensorType.INT8, input_shape, isinput=True)
    w = subgraph.create_tensor("weights", TensorType.INT8, weight_shape)
    b = subgraph.create_tensor("bso", TensorType.INT16, bso_shape)

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


def build_non_input_pad(subgraph=None, *, input_shape, paddings):
    model = build_pad(subgraph, input_shape=input_shape, paddings=paddings)
    subgraph = subgraph or model.subgraphs[0]

    build_abs(subgraph, input_shape=input_shape, tensor_type=TensorType.INT8)

    pad1, abs1 = subgraph.operators[:2]
    _glue_ops(abs1, pad1)

    return model


def build_reshape(
    subgraph=None,
    *,
    input_shape,
    output_shape,
    add_batch_dim=False,
    input_shape_tensor=True,
):

    if add_batch_dim:
        # Prepend dims with batch dimension 1
        input_shape = [1, *input_shape]

    assert 0 < len(output_shape) < 5

    assert np.prod(input_shape) == np.prod(output_shape), "Inconsistant shapes"

    subgraph = subgraph or XCOREModel().create_subgraph()

    tin = subgraph.create_tensor(
        "original_shape", TensorType.INT8, input_shape, isinput=True
    )
    tout = subgraph.create_tensor("reshaped", tin.type, output_shape, isoutput=True)

    if input_shape_tensor:
        p = subgraph.create_tensor("shape", TensorType.INT32, shape=[len(output_shape)])
        p.buffer.data = np.int32(output_shape)
        inputs = [tin, p]
    else:
        inputs = [tin]

    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.RESHAPE),
        inputs=inputs,
        outputs=[tout],
        builtin_options={"new_shape": output_shape},
    )
    return subgraph.model


def build_fc_with_reshape(
    subgraph=None, *, input_shape, fc_outputs, reshaped_input_shape
):
    model = build_reshape(
        subgraph,
        input_shape=input_shape,
        output_shape=reshaped_input_shape,
        add_batch_dim=False,
    )
    subgraph = subgraph or model.subgraphs[0]

    build_fc(
        subgraph,
        outputs=fc_outputs,
        input_shape=reshaped_input_shape,
        add_batch_dim=False,
    )

    reshape1, fc1 = subgraph.operators[:2]

    _glue_ops(reshape1, fc1)

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
