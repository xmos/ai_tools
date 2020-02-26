# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ReplaceSingleinDeepoutDepthwiseConv2DPass

from ...model_builders import build_depthwise_conv2d as build_model
from ..conftest import (
    _test_non_matching_stride_w,
    _test_non_matching_stride_h,
    _test_non_matching_output_channels,
    _test_non_matching_kernel_height,
    _test_non_matching_kernel_width,
    _test_non_matching_input_channels,
    _test_non_matching_types
)
from .test_ReplaceShallowinDeepoutConv2DPass import non_matching_kernel_width


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

MATCHING_INPUT_CHANNELS = [1]

NON_MATCHING_INPUT_CHANNELS = [2, 3, 4, 5, 8, 16, 32]


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def trf_pass():
    return ReplaceSingleinDeepoutDepthwiseConv2DPass()


@pytest.fixture(params=MATCHING_INPUT_CHANNELS)
def input_channels(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_INPUT_CHANNELS)
def non_matching_input_channels(request):
    return request.param


@pytest.fixture()
def weight_shape(output_channels, kernel_height, kernel_width, input_channels):
    return [output_channels, kernel_height, kernel_width, input_channels]


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    return build_model(weight_shape=weight_shape, input_size=input_size,
                       padding=padding, strides=strides)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


def test_non_matching_stride_w(trf_pass, model, non_matching_stride_w):
    _test_non_matching_stride_w(trf_pass, model, non_matching_stride_w)


def test_non_matching_stride_h(trf_pass, model, non_matching_stride_h):
    _test_non_matching_stride_h(trf_pass, model, non_matching_stride_h)


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
