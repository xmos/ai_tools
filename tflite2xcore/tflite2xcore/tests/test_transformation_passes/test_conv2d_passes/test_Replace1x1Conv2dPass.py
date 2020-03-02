# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import Replace1x1Conv2dPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_conv2d as build_model
)
from .conftest import (
    _pytest_generate_tests,
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
        "kernel_width": [1],
        "non_matching_kernel_width": [2, 3, 5],
        "kernel_height": [1],
        "non_matching_kernel_height": [2, 3, 5],
        "input_channels": [4, 8, 16, 32],
        "non_matching_input_channels": [3, 9, 15],
        "output_channels": [4, 8, 16, 32],
        "non_matching_output_channels": [3, 9, 15]
    },
    "smoke": {
        "kernel_width": [1],
        "non_matching_kernel_width": [3, 5],
        "kernel_height": [1],
        "non_matching_kernel_height": [3, 5],
        "input_channels": [4, 32],
        "non_matching_input_channels": [3, 9],
        "output_channels": [4, 32],
        "non_matching_output_channels": [3, 9]
    }
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    _pytest_generate_tests(metafunc, PARAMS)


@pytest.fixture()
def trf_pass():
    return Replace1x1Conv2dPass()


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
