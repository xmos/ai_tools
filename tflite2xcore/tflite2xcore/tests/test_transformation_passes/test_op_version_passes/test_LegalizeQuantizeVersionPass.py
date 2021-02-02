# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from typing import Tuple
from copy import deepcopy

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import BuiltinOpCodes
from tflite2xcore.transformation_passes import LegalizeQuantizeVersionPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_quantize

from .conftest import (
    PARAMS,
    _test_non_matching_params,
    _make_name_type_pairs,
    NON_INT8_TEST_TYPES,
    NON_FLOAT32_TEST_TYPES,
    test_matching_params,
    test_non_matching_tensors,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

_NON_MATCHING_TENSORS = list(
    _make_name_type_pairs("input", NON_FLOAT32_TEST_TYPES)
) + list(_make_name_type_pairs("output_quantized", NON_INT8_TEST_TYPES))

for params in PARAMS.values():
    params.update(
        {"non_matching_version": [1, 3], "non_matching_tensors": _NON_MATCHING_TENSORS}
    )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def model(input_shape: Tuple[int, int, int]) -> XCOREModel:
    model = build_quantize(input_shape=input_shape)
    model.subgraphs[0].operators[0].operator_code.version = 2
    return model


@pytest.fixture()
def trf_pass() -> LegalizeQuantizeVersionPass:
    return LegalizeQuantizeVersionPass()


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(model: XCOREModel, trf_pass: LegalizeQuantizeVersionPass) -> None:
    subgraph = model.subgraphs[0]
    trf_pass.mutate(subgraph.operators[0])
    subgraph.sanity_check()

    assert len(subgraph.operators) == 1
    assert len(subgraph.tensors) == 2

    op_code = subgraph.operators[0].operator_code
    assert op_code.code is BuiltinOpCodes.QUANTIZE
    assert op_code.version == 1


def test_non_matching_version(
    trf_pass: LegalizeQuantizeVersionPass,
    input_shape: Tuple[int, int, int],
    non_matching_version: int,
) -> None:
    model = build_quantize(input_shape=input_shape)
    model.subgraphs[0].operators[0].operator_code.version = non_matching_version
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
