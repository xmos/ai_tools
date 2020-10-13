# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import itertools
from typing import List, Dict, Iterator, Tuple, Any

from tflite2xcore.transformation_passes import ModelTransformationPass
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType, ValidOpCodes
from tflite2xcore.converter import CleanupManager


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------


def _make_name_type_pairs(
    name: str, types: List[TensorType]
) -> Iterator[Dict[str, TensorType]]:
    for n, t in zip(itertools.cycle([name]), types):
        yield {n: t}


def _test_non_matching_params(
    trf_pass: ModelTransformationPass, model: XCOREModel, *, op_idx: int = -1
) -> None:
    assert not trf_pass.match(model.subgraphs[0].operators[op_idx])


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------


NON_INT8_TEST_TYPES = [
    TensorType.UINT8,
    TensorType.INT32,
    TensorType.FLOAT32,
    TensorType.INT16,
]

NON_INT32_TEST_TYPES = [
    TensorType.INT8,
    TensorType.UINT8,
    TensorType.INT16,
    TensorType.FLOAT32,
]

NON_FLOAT32_TEST_TYPES = [
    TensorType.INT8,
    TensorType.INT16,
    TensorType.INT32,
    TensorType.UINT8,
]

ParamsType = Dict[str, Dict[str, List[Any]]]

PARAMS = {
    "extended": {
        "input_height": [7, 9, 17, 20, 32],
        "input_width": [7, 9, 17, 20, 32],
        "input_channels": [4, 8, 16, 32, 36, 64],
    },
    "default": {
        "input_height": [9, 20],
        "input_width": [7, 17],
        "input_channels": [4, 8, 16, 32],
    },
    "smoke": {
        "input_height": [9, 20],
        "input_width": [7, 17],
        "input_channels": [4, 32],
    },
}  # type: ParamsType


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def strides(stride_h: int, stride_w: int) -> Tuple[int, int]:
    return (stride_h, stride_w)


@pytest.fixture()
def input_size(input_height: int, input_width: int) -> Tuple[int, int]:
    return (input_height, input_width)


@pytest.fixture()
def input_shape(
    input_size: Tuple[int, int], input_channels: int
) -> Tuple[int, int, int]:
    return (*input_size, input_channels)


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_matching_params(
    trf_pass: ModelTransformationPass, model: XCOREModel, *, op_idx: int = -1
) -> None:
    assert trf_pass.match(model.subgraphs[0].operators[op_idx])


def test_non_matching_tensors(
    trf_pass: ModelTransformationPass,
    model: XCOREModel,
    non_matching_tensors: Dict[str, TensorType],
) -> None:
    subgraph = model.subgraphs[0]
    for name, type_ in non_matching_tensors.items():
        subgraph.get_tensor(name).type = type_
    _test_non_matching_params(trf_pass, model)


def test_replace_mutate(
    trf_pass: ModelTransformationPass, model: XCOREModel, new_opcode: ValidOpCodes
) -> None:
    # run replacement pass
    trf_pass.run(model)
    model.sanity_check()

    # clean up dangling op
    CleanupManager(model).run_passes()
    model.sanity_check()

    # check new op
    op = model.subgraphs[0].operators[-1]
    assert op.operator_code.code is new_opcode
