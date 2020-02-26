# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import TensorType


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

MATCHING_INPUT_HEIGHT = [9, 20]
MATCHING_INPUT_WIDTH = [7, 17]
MATCHING_OUTPUT_CHANNELS = [16, 32]
MATCHING_KERNEL_HEIGHT = [1, 3, 5, 7]
MATCHING_KERNEL_WIDTH = MATCHING_KERNEL_HEIGHT
MATCHING_PADDING = ['SAME', 'VALID']

NON_MATCHING_STRIDE_W = [2, 3]
NON_MATCHING_STRIDE_H = NON_MATCHING_STRIDE_W
NON_MATCHING_OUTPUT_CHANNELS = [8, 24, 17, 63]
NON_MATCHING_KERNEL_HEIGHT = [2, 4, 6]
NON_MATCHING_TENSORS = [
    ('input', TensorType.INT16), ('input', TensorType.INT32),
    ('weights', TensorType.INT16), ('weights', TensorType.INT32),
    ('biases', TensorType.INT8), ('biases', TensorType.INT16),
    ('output', TensorType.INT16), ('output', TensorType.INT32)
]


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------

def _test_non_matching_stride_w(trf_pass, model, non_matching_stride_w):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_w'] = non_matching_stride_w
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


def _test_non_matching_stride_h(trf_pass, model, non_matching_stride_h):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_h'] = non_matching_stride_h
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


def _test_non_matching_dim(trf_pass, build_model,
                           weight_shape, input_size, padding):
    model = build_model(weight_shape=weight_shape, input_size=input_size, padding=padding)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


def _test_non_matching_output_channels(trf_pass, build_model,
                                       weight_shape, input_size, padding,
                                       non_matching_output_channels):
    weight_shape[0] = non_matching_output_channels
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding)


def _test_non_matching_kernel_height(trf_pass, build_model,
                                     weight_shape, input_size, padding,
                                     non_matching_kernel_height):
    weight_shape[1] = non_matching_kernel_height
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding)


def _test_non_matching_kernel_width(trf_pass, build_model,
                                    weight_shape, input_size, padding,
                                    non_matching_kernel_width):
    weight_shape[2] = non_matching_kernel_width
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding)


def _test_non_matching_input_channels(trf_pass, build_model,
                                      weight_shape, input_size, padding,
                                      non_matching_input_channels):
    weight_shape[3] = non_matching_input_channels
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding)


def _test_non_matching_types(trf_pass, model, non_matching_tensors):
    subgraph = model.subgraphs[0]
    subgraph.get_tensor(non_matching_tensors[0]).type = non_matching_tensors[1]
    assert not trf_pass.match(subgraph.operators[-1])


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture(params=MATCHING_INPUT_HEIGHT)
def input_height(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_WIDTH)
def input_width(request):
    return request.param


@pytest.fixture()
def input_size(input_height, input_width):
    return [input_height, input_width]


@pytest.fixture(params=MATCHING_OUTPUT_CHANNELS)
def output_channels(request):
    return request.param


@pytest.fixture(params=MATCHING_KERNEL_HEIGHT)
def kernel_height(request):
    return request.param


@pytest.fixture(params=MATCHING_KERNEL_WIDTH)
def kernel_width(request):
    return request.param


@pytest.fixture(params=MATCHING_PADDING)
def padding(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_STRIDE_W)
def non_matching_stride_w(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_STRIDE_H)
def non_matching_stride_h(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_OUTPUT_CHANNELS)
def non_matching_output_channels(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_KERNEL_HEIGHT)
def non_matching_kernel_height(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_TENSORS)
def non_matching_tensors(request):
    return request.param
