# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import TensorType
from tflite2xcore.transformation_passes import Replace1x1Conv2dPass

from ..model_builders import build_conv2d as build_model
from .conftest import (
    _test_non_matching_stride_w,
    _test_non_matching_stride_h,
    _test_non_matching_output_channels,
    _test_non_matching_kernel_height,
    _test_non_matching_kernel_width,
    _test_non_matching_input_channels,
    _test_non_matching_types
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

MATCHING_KERNEL_HEIGHT = [1]
MATCHING_KERNEL_WIDTH = MATCHING_KERNEL_HEIGHT
MATCHING_INPUT_CHANNELS = [4, 8, 16, 32]
MATCHING_OUTPUT_CHANNELS = MATCHING_INPUT_CHANNELS
MATCHING_STRIDE_W = [1, 2, 3]
MATCHING_STRIDE_H = MATCHING_STRIDE_W

NON_MATCHING_KERNEL_HEIGHT = [2, 3, 5]
NON_MATCHING_KERNEL_WIDTH = NON_MATCHING_KERNEL_HEIGHT
NON_MATCHING_INPUT_CHANNELS = [3, 9, 15]
NON_MATCHING_OUTPUT_CHANNELS = NON_MATCHING_INPUT_CHANNELS


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def trf_pass():
    return Replace1x1Conv2dPass()


@pytest.fixture(params=MATCHING_KERNEL_HEIGHT)
def kernel_height(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_KERNEL_HEIGHT)
def non_matching_kernel_height(request):
    return request.param


@pytest.fixture(params=MATCHING_KERNEL_WIDTH)
def kernel_width(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_KERNEL_WIDTH)
def non_matching_kernel_width(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_CHANNELS)
def input_channels(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_INPUT_CHANNELS)
def non_matching_input_channels(request):
    return request.param


@pytest.fixture(params=MATCHING_OUTPUT_CHANNELS)
def output_channels(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_OUTPUT_CHANNELS)
def non_matching_output_channels(request):
    return request.param


@pytest.fixture()
def weight_shape(output_channels, kernel_height, kernel_width, input_channels):
    return [output_channels, kernel_height, kernel_width, input_channels]


@pytest.fixture(params=MATCHING_STRIDE_W)
def stride_w(request):
    return request.param


@pytest.fixture(params=MATCHING_STRIDE_H)
def stride_h(request):
    return request.param


@pytest.fixture()
def strides(stride_h, stride_w):
    return (stride_h, stride_w)


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        padding=padding, strides=strides)
    return model


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


def test_non_matching_output_channels(trf_pass,
                                      weight_shape, input_size, padding, strides,
                                      non_matching_output_channels):
    _test_non_matching_output_channels(trf_pass, build_model,
                                       weight_shape, input_size, padding, strides,
                                       non_matching_output_channels)


def test_non_matching_kernel_height(trf_pass,
                                    weight_shape, input_size, padding, strides,
                                    non_matching_kernel_height):
    _test_non_matching_kernel_height(trf_pass, build_model,
                                     weight_shape, input_size, padding, strides,
                                     non_matching_kernel_height)


def test_non_matching_kernel_width(trf_pass,
                                   weight_shape, input_size, padding, strides,
                                   non_matching_kernel_width):
    _test_non_matching_kernel_width(trf_pass, build_model,
                                    weight_shape, input_size, padding, strides,
                                    non_matching_kernel_width)


def test_non_matching_input_channels(trf_pass,
                                     weight_shape, input_size, padding, strides,
                                     non_matching_input_channels):
    _test_non_matching_input_channels(trf_pass, build_model,
                                      weight_shape, input_size, padding, strides,
                                      non_matching_input_channels)


def test_non_matching_types(trf_pass, model, non_matching_tensors):
    _test_non_matching_types(trf_pass, model, non_matching_tensors)


if __name__ == "__main__":
    pytest.main()
