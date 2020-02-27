# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import TensorType
from tflite2xcore.transformation_passes import ReplaceDeepinDeepoutConv2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_conv2d as build_model
)
from . import conftest
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
        "non_matching_kernel_width": conftest.PARAMS["default"]["non_matching_kernel_height"],
        "input_channels": [32, 64],
        "non_matching_input_channels": [8, 16, 33, 48]
    },
    "smoke": {
        "non_matching_kernel_width": conftest.PARAMS["smoke"]["non_matching_kernel_height"],
        "input_channels": [32],
        "non_matching_input_channels": [16]
    }
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    _pytest_generate_tests(metafunc, PARAMS)


@pytest.fixture()
def trf_pass():
    return ReplaceDeepinDeepoutConv2DPass()


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
