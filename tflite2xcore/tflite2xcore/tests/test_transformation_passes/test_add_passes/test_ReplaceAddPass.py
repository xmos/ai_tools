# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest
from typing import Tuple, Optional


from tflite2xcore.transformation_passes import ReplaceAddPass
from tflite2xcore.xcore_schema import (
    XCOREModel,
    Subgraph,
    BuiltinOpCodes,
    OperatorCode,
    TensorType,
    XCOREOpCodes,
    Subgraph,
)
from . import test_replace_mutate as _test_replace_mutate


#  ----------------------------------------------------------------------------
#                              HELPERS
#  ----------------------------------------------------------------------------


def build_add(
    subgraph: Optional[Subgraph] = None,
    *,
    input_shape: Tuple[int, ...],
    tensor_type: TensorType
) -> XCOREModel:
    subgraph = subgraph or Subgraph(model=XCOREModel())
    input_tensor_0 = subgraph.create_tensor(
        "input_0",
        tensor_type,
        input_shape,
        isinput=True,
        quantization={"scale": [0.35], "zero_point": [0]},
    )
    input_tensor_1 = subgraph.create_tensor(
        "input_1",
        tensor_type,
        input_shape,
        isinput=True,
        quantization={"scale": [0.15], "zero_point": [2]},
    )
    output_tensor = subgraph.create_tensor(
        "output",
        tensor_type,
        input_shape,
        isoutput=True,
        quantization={"scale": [0.25], "zero_point": [1]},
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
    return XCOREOpCodes.XC_add_8


@pytest.fixture()
def model(input_shape: Tuple[int, int, int], tensor_type: TensorType) -> XCOREModel:
    return build_add(input_shape=input_shape, tensor_type=tensor_type)


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_matching_params(trf_pass: ReplaceAddPass, model: XCOREModel) -> None:
    assert trf_pass.match(model.subgraphs[0].operators[0])


def test_non_matching_tensor_type(
    trf_pass: ReplaceAddPass, non_matching_tensor_type: TensorType, model: XCOREModel
) -> None:
    model.subgraphs[0].get_tensor("input_1").type = TensorType
    assert not trf_pass.match(model.subgraphs[0].operators[0])


def test_non_matching_tensor_shape(trf_pass: ReplaceAddPass, model: XCOREModel) -> None:
    current_shape = model.subgraphs[0].get_tensor("input_1").shape
    new_shape = (current_shape[0] + 1, *current_shape[1:])
    model.subgraphs[0].get_tensor("input_1").shape = new_shape
    assert not trf_pass.match(model.subgraphs[0].operators[0])


def test_mutate(
    trf_pass: ReplaceAddPass, model: XCOREModel, new_opcode: XCOREOpCodes
) -> None:
    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    _test_replace_mutate(trf_pass, model, new_opcode)

    # check operators
    subgraph = model.subgraphs[0]
    operators = subgraph.operators
    assert len(operators) == 1
    op = operators[0]

    # check tensors
    assert len(op.inputs) == 3
    assert len(op.outputs) == 1
    assert op.inputs[2].shape == (6,)
    assert op.inputs[2].type is TensorType.INT32


if __name__ == "__main__":
    pytest.main()
