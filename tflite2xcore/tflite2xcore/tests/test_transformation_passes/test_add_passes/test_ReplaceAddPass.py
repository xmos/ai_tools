# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
from typing import Tuple


from tflite2xcore.transformation_passes import ReplaceAddPass
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import (
    BuiltinOpCodes,
    OperatorCode,
    TensorType,
    XCOREOpCodes,
    Subgraph,
)
from . import test_replace_mutate as test_mutate


#  ----------------------------------------------------------------------------
#                              HELPERS
#  ----------------------------------------------------------------------------


def build_add(
    subgraph: Subgraph = None,
    *,
    input_shape: Tuple[int, int, int, int],
    tensor_type: TensorType
) -> XCOREModel:
    subgraph = subgraph or XCOREModel().create_subgraph()
    input_tensor_0 = subgraph.create_tensor(
        "input_0", tensor_type, input_shape, isinput=True
    )
    input_tensor_1 = subgraph.create_tensor(
        "input_1", tensor_type, input_shape, isinput=True
    )
    output_tensor = subgraph.create_tensor(
        "output", tensor_type, input_shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.ADD),
        inputs=[input_tensor_0, input_tensor_1],
        outputs=[output_tensor],
    )
    return subgraph.model


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = {
    "default": {
        "tensor_type": [TensorType.INT8],
        "non_matching_tensor_type": [TensorType.INT16, TensorType.FLOAT32],
        "input_height": [9, 20],
        "input_width": [7, 17],
        "input_channels": [4, 16, 32],
    }
}

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ReplaceAddPass:
    return ReplaceAddPass()


@pytest.fixture()
def new_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_add


@pytest.fixture()
def model(input_shape: Tuple[int, int, int], tensor_type: TensorType) -> XCOREModel:
    return build_add(input_shape=input_shape, tensor_type=tensor_type)


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_matching_params(trf_pass: ReplaceAddPass, model: XCOREModel) -> None:
    assert trf_pass.match(model.subgraphs[0].operators[0])


def test_non_matching_tensor_type(
    trf_pass: ReplaceAddPass, non_matching_tensor_type: TensorType
) -> None:
    model = build_add(input_shape=(1, 1, 1, 1), tensor_type=non_matching_tensor_type)
    assert not trf_pass.match(model.subgraphs[0].operators[0])


def test_non_matching_tensor_shape(trf_pass: ReplaceAddPass, model: XCOREModel) -> None:
    model.subgraphs[0].get_tensor("input_1").shape = (2, 2, 2, 2)
    assert not trf_pass.match(model.subgraphs[0].operators[0])


if __name__ == "__main__":
    pytest.main()
