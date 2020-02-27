# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.tests.conftest import _pytest_generate_tests
from tflite2xcore.xcore_model import TensorType


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = {
    "default": {
        "input_height": [9, 20],
        "input_width": [7, 17],
        "padding": ['SAME', 'VALID'],
        "non_matching_tensors": [
            ('input', TensorType.INT16), ('input', TensorType.INT32),
            ('weights', TensorType.INT16), ('weights', TensorType.INT32),
            ('biases', TensorType.INT8), ('biases', TensorType.INT16),
            ('output', TensorType.INT16), ('output', TensorType.INT32)
        ]
    },
    "smoke": {
        "input_height": [9, 20],
        "input_width": [7, 17],
        "padding": ['SAME', 'VALID'],
        "non_matching_tensors": [
            ('input', TensorType.INT16),
            ('weights', TensorType.INT16),
            ('biases', TensorType.INT8),
            ('output', TensorType.INT16)
        ]
    }
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    _pytest_generate_tests(metafunc, PARAMS)


@pytest.fixture()
def weight_shape(output_channels, kernel_height, kernel_width, input_channels):
    return [output_channels, kernel_height, kernel_width, input_channels]


@pytest.fixture()
def input_size(input_height, input_width):
    return [input_height, input_width]


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------


def _test_non_matching_dim(trf_pass, build_model,
                           weight_shape, input_size, padding, strides):
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        padding=padding, strides=strides)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


def _test_non_matching_output_channels(trf_pass, build_model,
                                       weight_shape, input_size, padding, strides,
                                       non_matching_output_channels):
    weight_shape[0] = non_matching_output_channels
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


def _test_non_matching_kernel_height(trf_pass, build_model,
                                     weight_shape, input_size, padding, strides,
                                     non_matching_kernel_height):
    weight_shape[1] = non_matching_kernel_height
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


def _test_non_matching_kernel_width(trf_pass, build_model,
                                    weight_shape, input_size, padding, strides,
                                    non_matching_kernel_width):
    weight_shape[2] = non_matching_kernel_width
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


def _test_non_matching_input_channels(trf_pass, build_model,
                                      weight_shape, input_size, padding, strides,
                                      non_matching_input_channels):
    weight_shape[3] = non_matching_input_channels
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


def _test_non_matching_types(trf_pass, model, non_matching_tensors):
    subgraph = model.subgraphs[0]
    subgraph.get_tensor(non_matching_tensors[0]).type = non_matching_tensors[1]
    assert not trf_pass.match(subgraph.operators[-1])
