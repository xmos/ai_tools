# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_schema import TensorType

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

_NON_MATCHING_TENSORS = [
    {"input": TensorType.UINT8},
    {"input": TensorType.INT32},
    {"input": TensorType.FLOAT32},
    {"input": TensorType.INT16},
    {"weights": TensorType.UINT8},
    {"weights": TensorType.INT32},
    {"weights": TensorType.FLOAT32},
    {"weights": TensorType.INT16},
    {"biases": TensorType.INT8},
    {"biases": TensorType.UINT8},
    {"biases": TensorType.FLOAT32},
    {"biases": TensorType.INT16},
    {"output": TensorType.UINT8},
    {"output": TensorType.INT32},
    {"output": TensorType.FLOAT32},
    {"output": TensorType.INT16},
]

PARAMS = {
    "extended": {
        "input_height": [7, 9, 17, 20, 32],
        "input_width": [7, 9, 17, 20, 32],
        "input_channels": [4, 8, 16, 32, 36, 64],
        "non_matching_tensors": _NON_MATCHING_TENSORS,
    },
    "default": {
        "input_height": [9, 20],
        "input_width": [7, 17],
        "input_channels": [4, 8, 16, 32],
        "non_matching_tensors": _NON_MATCHING_TENSORS[::2],
    },
    "smoke": {
        "input_height": [9, 20],
        "input_width": [7, 17],
        "input_channels": [4, 32],
        "non_matching_tensors": _NON_MATCHING_TENSORS[::4],
    },
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def strides(stride_h, stride_w):
    return (stride_h, stride_w)


@pytest.fixture()
def input_size(input_height, input_width):
    return [input_height, input_width]


@pytest.fixture()
def input_shape(input_size, input_channels):
    return [*input_size, input_channels]


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------


def _test_non_matching_params(trf_pass, model):
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


def _test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_matching_params(trf_pass, model):
    _test_matching_params(trf_pass, model)


def test_non_matching_tensors(trf_pass, model, non_matching_tensors):
    subgraph = model.subgraphs[0]
    for name, type_ in non_matching_tensors.items():
        subgraph.get_tensor(name).type = type_
    _test_non_matching_params(trf_pass, model)
