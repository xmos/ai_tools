# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.xcore_model import TensorType

from ..conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update({
    "kernel_height": [2, 3, 5, 7],
    "kernel_width": [2, 3, 5, 7],
    "non_matching_input_channels": [3, 9, 15],
    "output_channels": [4, 8, 16, 32],
    "non_matching_output_channels": [3, 9, 15],
    "padding": ['SAME', 'VALID'],
    "stride_h": [1],  # TODO: this should be extended after the conv2d improvements
    "non_matching_stride_h": [2, 3],  # TODO: this should be removed after the conv2d improvements
    "stride_w": [1],  # TODO: this should be extended after the conv2d improvements
    "non_matching_stride_w": [2, 3],  # TODO: this should be removed after the conv2d improvements
    "non_matching_tensors": [
        ('input', TensorType.INT16), ('input', TensorType.INT32),
        ('weights', TensorType.INT16), ('weights', TensorType.INT32),
        ('biases', TensorType.INT8), ('biases', TensorType.INT16),
        ('output', TensorType.INT16), ('output', TensorType.INT32)
    ]
})

PARAMS["smoke"].update({
    "kernel_height": [2, 3],
    "kernel_width": [2, 3],
    "non_matching_input_channels": [3, 9],
    "output_channels": [4, 32],
    "non_matching_output_channels": [3, 9],
    "padding": ['SAME', 'VALID'],
    "stride_h": [1],  # TODO: this should be extended after the conv2d improvements
    "non_matching_stride_h": [2],  # TODO: this should be removed after the conv2d improvements
    "stride_w": [1],  # TODO: this should be extended after the conv2d improvements
    "non_matching_stride_w": [2],  # TODO: this should be removed after the conv2d improvements
    "non_matching_tensors": [
        ('input', TensorType.INT16),
        ('weights', TensorType.INT16),
        ('biases', TensorType.INT8),
        ('output', TensorType.INT16)
    ]
})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def weight_shape(output_channels, kernel_height, kernel_width, input_channels):
    return [output_channels, kernel_height, kernel_width, input_channels]


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------

def test_non_matching_output_channels(trf_pass, build_model,
                                      weight_shape, input_size, padding, strides,
                                      non_matching_output_channels):
    weight_shape[0] = non_matching_output_channels
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        padding=padding, strides=strides)
    _test_non_matching_params(trf_pass, model)


def test_non_matching_kernel_height(trf_pass, build_model,
                                    weight_shape, input_size, padding, strides,
                                    non_matching_kernel_height):
    weight_shape[1] = non_matching_kernel_height
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        padding=padding, strides=strides)
    _test_non_matching_params(trf_pass, model)


def test_non_matching_kernel_width(trf_pass, build_model,
                                   weight_shape, input_size, padding, strides,
                                   non_matching_kernel_width):
    weight_shape[2] = non_matching_kernel_width
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        padding=padding, strides=strides)
    _test_non_matching_params(trf_pass, model)


def test_non_matching_input_channels(trf_pass, build_model,
                                     weight_shape, input_size, padding, strides,
                                     non_matching_input_channels):
    weight_shape[3] = non_matching_input_channels
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        padding=padding, strides=strides)
    _test_non_matching_params(trf_pass, model)


def test_non_matching_types(trf_pass, model, non_matching_tensors):
    subgraph = model.subgraphs[0]
    subgraph.get_tensor(non_matching_tensors[0]).type = non_matching_tensors[1]
    _test_non_matching_params(trf_pass, model)


def test_non_matching_stride_w(trf_pass, model, non_matching_stride_w):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_w'] = non_matching_stride_w
    _test_non_matching_params(trf_pass, model)


def test_non_matching_stride_h(trf_pass, model, non_matching_stride_h):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_h'] = non_matching_stride_h
    _test_non_matching_params(trf_pass, model)
