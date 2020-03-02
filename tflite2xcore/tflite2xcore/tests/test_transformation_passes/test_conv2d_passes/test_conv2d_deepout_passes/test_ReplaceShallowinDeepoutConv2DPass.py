# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ReplaceShallowinDeepoutConv2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_conv2d as build_model
from .conftest import (
    _pytest_generate_tests,
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

PARAMS = {
    "default": {
        "non_matching_kernel_width": [2, 4, 6, 9],
        "input_channels": [1, 2, 3, 4],
        "non_matching_input_channels": [5, 16, 32]
    },
    "smoke": {
        "non_matching_kernel_width": [2, 9],
        "input_channels": [1, 4],
        "non_matching_input_channels": [5, 32]
    }
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    _pytest_generate_tests(metafunc, PARAMS)


@pytest.fixture()
def trf_pass():
    return ReplaceShallowinDeepoutConv2DPass()


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
