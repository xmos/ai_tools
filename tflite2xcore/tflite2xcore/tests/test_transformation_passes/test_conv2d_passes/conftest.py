# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import TensorType


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = {
    "default": {
        "input_height": [9, 20],
        "input_width": [7, 17],
        "kernel_height": [2, 3, 5, 7],
        "kernel_width": [2, 3, 5, 7],
        "input_channels": [4, 8, 16, 32],
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
    },
    "smoke": {
        "input_height": [9, 20],
        "input_width": [7, 17],
        "kernel_height": [2, 3],
        "kernel_width": [2, 3],
        "input_channels": [4, 32],
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
    }
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def strides(stride_h, stride_w):
    return (stride_h, stride_w)


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


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------

def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


def test_non_matching_output_channels(trf_pass, build_model,
                                      weight_shape, input_size, padding, strides,
                                      non_matching_output_channels):
    weight_shape[0] = non_matching_output_channels
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


def test_non_matching_kernel_height(trf_pass, build_model,
                                    weight_shape, input_size, padding, strides,
                                    non_matching_kernel_height):
    weight_shape[1] = non_matching_kernel_height
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


def test_non_matching_kernel_width(trf_pass, build_model,
                                   weight_shape, input_size, padding, strides,
                                   non_matching_kernel_width):
    weight_shape[2] = non_matching_kernel_width
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


def test_non_matching_input_channels(trf_pass, build_model,
                                     weight_shape, input_size, padding, strides,
                                     non_matching_input_channels):
    weight_shape[3] = non_matching_input_channels
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


def test_non_matching_types(trf_pass, model, non_matching_tensors):
    subgraph = model.subgraphs[0]
    subgraph.get_tensor(non_matching_tensors[0]).type = non_matching_tensors[1]
    assert not trf_pass.match(subgraph.operators[-1])


def test_non_matching_stride_w(trf_pass, model, non_matching_stride_w):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_w'] = non_matching_stride_w
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


def test_non_matching_stride_h(trf_pass, model, non_matching_stride_h):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_h'] = non_matching_stride_h
    assert not trf_pass.match(model.subgraphs[0].operators[-1])
