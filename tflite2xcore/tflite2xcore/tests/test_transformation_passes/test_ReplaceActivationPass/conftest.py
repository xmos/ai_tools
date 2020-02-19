# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.operator_codes import XCOREOpCodes
from tflite2xcore.xcore_model import TensorType

MATCHING_INPUT_CHANNELS = [1, 3, 4, 16, 32]
MATCHING_INPUT_HEIGHT = [1, 3, 8]
MATCHING_INPUT_WIDTH = MATCHING_INPUT_HEIGHT

NON_MATCHING_INPUT_TYPE = [
    TensorType.INT16, TensorType.INT32, TensorType.UINT8, TensorType.FLOAT32
]
NON_MATCHING_OUTPUT_TYPE = NON_MATCHING_INPUT_TYPE


# helpers

def _test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[0])


def _test_non_matching_input_type(trf_pass, model, non_matching_input_type):
    op = model.subgraphs[0].operators[0]
    op.inputs[0].type = non_matching_input_type
    assert not trf_pass.match(op)


def _test_non_matching_output_type(trf_pass, model, non_matching_output_type):
    op = model.subgraphs[0].operators[0]
    op.outputs[0].type = non_matching_output_type
    assert not trf_pass.match(op)


def _test_mutate(trf_pass, model):
    trf_pass.run(model)
    subgraph = model.subgraphs[0]
    op = subgraph.operators[-1]
    assert op.operator_code.code == XCOREOpCodes.XC_lookup_8

    # check input/output tensors
    tin = subgraph.get_tensor('input')
    tout = subgraph.get_tensor('output')

    assert len(subgraph.operators) == 1
    assert len(subgraph.tensors) == 3
    assert tin in subgraph.inputs and tin not in subgraph.outputs
    assert tout in subgraph.outputs and tout not in subgraph.inputs

    # check LUT
    lut_tensor = op.inputs[1]
    assert len(lut_tensor.buffer.data) == 256
    assert lut_tensor.shape == [256]


# fixtures

@pytest.fixture(params=MATCHING_INPUT_CHANNELS)
def input_channels(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_HEIGHT)
def input_height(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_WIDTH)
def input_width(request):
    return request.param


@pytest.fixture()
def input_shape(input_height, input_width, input_channels):
    return [input_height, input_width, input_channels]


@pytest.fixture(params=NON_MATCHING_INPUT_TYPE)
def non_matching_input_type(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_OUTPUT_TYPE)
def non_matching_output_type(request):
    return request.param