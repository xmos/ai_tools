# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy
import pytest

from pytest_cases import pytest_fixture_plus, pytest_parametrize_plus, fixture_ref
from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import ReplaceDeepinAnyoutFullyConnectedOutputPass

from .fully_connected_composite_test import build_fc as build_model

from .test_ReplaceDeepinDeepoutConv2DPass import (
    NON_MATCHING_TENSORS
)


# test case parameter definitions, TODO: refactor what's common here
MATCHING_OUTPUTS = [1, 2, 10, 15, 16, 17, 100]
MATCHING_INPUT_SIZE = [
    (1, 1, 32), (2, 2, 8), (4, 4, 2), (32,),
    (1, 2, 32), (4, 2, 8), (8, 8, 1), (64,)
]


NON_MATCHING_INPUT_SIZE = [
    (1, 1, 31), (2, 2, 7), (3, 4, 3), (33,),
    (2, 2, 15), (3, 3, 7), (9, 4, 6), (63,)
]


@pytest.fixture()
def trf_pass():
    return ReplaceDeepinAnyoutFullyConnectedOutputPass()


@pytest.fixture(params=MATCHING_OUTPUTS)
def outputs(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_SIZE)
def input_size(request):
    return request.param


@pytest.fixture()
def model(outputs, input_size):
    return build_model(outputs=outputs, input_size=input_size)


def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_size', NON_MATCHING_INPUT_SIZE)
def test_non_matching_input_size(trf_pass, outputs, input_size):
    model = build_model(outputs=outputs, input_size=input_size)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize(*NON_MATCHING_TENSORS)
def test_non_matching_types(trf_pass, model, tensor_name, new_type):
    subgraph = model.subgraphs[0]
    subgraph.get_tensor(tensor_name).type = new_type
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


if __name__ == "__main__":
    pytest.main()
