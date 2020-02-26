# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, XCOREOpCodes
from tflite2xcore.transformation_passes import ParallelizeDIDOPass


from .test_conv2d_passes.test_conv2d_deepout_passes.conftest import (
    output_channels, kernel_height, kernel_width,
    input_height, input_width, input_size,
    padding
)

from .test_conv2d_passes.test_conv2d_deepout_passes.test_ReplaceSingleinDeepoutDepthwiseConv2DPass import (
    input_channels, weight_shape, model
)


def build_DIDO(*, weight_shape, input_size, padding):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    height, width = input_size
    C_out, K_h, K_w, C_in = weight_shape

    input_shape = [1, height, width, C_in]
    tin = subgraph.create_tensor('input', TensorType.INT8, shape=input_shape, isinput=True)
    kernel_shape = [C_out // 16, K_h, K_w, C_in // 32, 16, 32]
    w = subgraph.create_tensor('weights', TensorType.INT8, shape=kernel_shape)
    bias_shape = [C_out // 16, 2, 16]
    b = subgraph.create_tensor('biases', TensorType.INT16, shape=bias_shape)
    scales_shape = bias_shape
    s = subgraph.create_tensor('scales', TensorType.INT16, shape=scales_shape)

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


@pytest.fixture()
def trf_pass(num_threads):
    return ParallelizeDIDOPass(num_threads=num_threads)


@pytest.fixture()
def model(weight_shape, input_size, padding):
    return build_DIDO(weight_shape=weight_shape, input_size=input_size, padding=padding)


@pytest.mark.parametrize('num_threads', [2])
def test_matching(trf_pass, model, num_threads):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('num_threads', [1])
def test_single_thread_identity(trf_pass, model, num_threads):
    op = model.subgraphs[0].operators[0]
    custom_options = deepcopy(op.custom_options)
    trf_pass.run(model)
    assert op.custom_options == custom_options


@pytest.mark.parametrize('num_threads', [2, 3, 4, 5])
def test_mutation(trf_pass, model, num_threads):
    op = model.subgraphs[0].operators[0]
    assert 'par_plan' not in op.custom_options
    trf_pass.run(model)
    assert 'par_plan' in op.custom_options


if __name__ == "__main__":
    pytest.main()
