# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
import itertools

from pytest_cases import pytest_fixture_plus, pytest_parametrize_plus, fixture_ref
from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import ReplaceShallowinDeepoutConv2DPass


# test case parameter definitions, TODO: refactor what's common here
from .test_ReplaceDeepinDeepoutConv2DPass import (
    MATCHING_OUTPUT_CHANNELS,
    MATCHING_KERNEL_HEIGHT,
    MATCHING_PADDING,

    NON_MATCHING_STRIDE_W,
    NON_MATCHING_STRIDE_H,
    NON_MATCHING_OUTPUT_CHANNELS,
    NON_MATCHING_KERNEL_HEIGHT,

    NON_MATCHING_TENSORS
)

MATCHING_KERNEL_WIDTH = [1, 3, 5, 7]
MATCHING_INPUT_CHANNELS = [1, 2, 3, 4]
MATCHING_INPUT_HEIGHT = [9, 12, 20]
MATCHING_INPUT_WIDTH = [7, 13, 17]

NON_MATCHING_KERNEL_WIDTH = [2, 4, 6, 9]
NON_MATCHING_INPUT_CHANNELS = [5, 8, 16, 32, 64]


# conv2d model builder, TODO: refactor what's common here
from .test_ReplaceDeepinDeepoutConv2DPass import (
    build_model,
)


@pytest.fixture()
def trf_pass():
    return ReplaceShallowinDeepoutConv2DPass()


@pytest.fixture(params=MATCHING_OUTPUT_CHANNELS)
def output_channels(request):
    return request.param


@pytest.fixture(params=MATCHING_KERNEL_HEIGHT)
def kernel_height(request):
    return request.param


@pytest.fixture(params=MATCHING_KERNEL_WIDTH)
def kernel_width(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_CHANNELS)
def input_channels(request):
    return request.param


@pytest.fixture()
def weight_shape(output_channels, kernel_height, kernel_width, input_channels):
    return [output_channels, kernel_height, kernel_width, input_channels]


@pytest.fixture(params=MATCHING_INPUT_HEIGHT)
def input_height(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_WIDTH)
def input_width(request):
    return request.param


@pytest.fixture()
def input_size(input_height, input_width):
    return [input_height, input_width]


@pytest.fixture(params=MATCHING_PADDING)
def padding(request):
    return request.param


@pytest.fixture()
def model(weight_shape, input_size, padding):
    return build_model(weight_shape=weight_shape, input_size=input_size, padding=padding)


def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('stride_w', NON_MATCHING_STRIDE_W)
def test_non_matching_stride_w(trf_pass, model, stride_w):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_w'] = stride_w
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('stride_h', NON_MATCHING_STRIDE_H)
def test_non_matching_stride_h(trf_pass, model, stride_h):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_h'] = stride_h
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('output_channels', NON_MATCHING_OUTPUT_CHANNELS)
def test_non_matching_output_channels(trf_pass, weight_shape, input_size, padding, output_channels):
    weight_shape[0] = output_channels
    model = build_model(weight_shape=weight_shape, input_size=input_size, padding=padding)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('kernel_height', NON_MATCHING_KERNEL_HEIGHT)
def test_non_matching_kernel_height(trf_pass, weight_shape, input_size, padding, kernel_height):
    weight_shape[1] = kernel_height
    model = build_model(weight_shape=weight_shape, input_size=input_size, padding=padding)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('kernel_width', NON_MATCHING_KERNEL_WIDTH)
def test_non_matching_kernel_width(trf_pass, weight_shape, input_size, padding, kernel_width):
    weight_shape[2] = kernel_width
    model = build_model(weight_shape=weight_shape, input_size=input_size, padding=padding)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_channels', NON_MATCHING_INPUT_CHANNELS)
def test_non_matching_input_channels(trf_pass, weight_shape, input_size, padding, input_channels):
    weight_shape[3] = input_channels
    model = build_model(weight_shape=weight_shape, input_size=input_size, padding=padding)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize(*NON_MATCHING_TENSORS)
def test_non_matching_types(trf_pass, model, tensor_name, new_type):
    subgraph = model.subgraphs[0]
    subgraph.get_tensor(tensor_name).type = new_type
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


if __name__ == "__main__":
    pytest.main()
