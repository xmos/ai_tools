# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import itertools
from copy import deepcopy
from typing import Tuple

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType, BuiltinOpCodes
from tflite2xcore.transformation_passes import RemoveRedundantInt8RequantizationPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_fc,
    build_abs,
    _glue_quantize,
)

from ..test_fully_connected_passes.conftest import PARAMS
from .conftest import (
    NON_INT8_TEST_TYPES,
    _make_name_type_pairs,
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_tensors,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

_NON_MATCHING_TENSORS = list(
    itertools.chain(
        _make_name_type_pairs("output_quantized", NON_INT8_TEST_TYPES),
        _make_name_type_pairs("output", NON_INT8_TEST_TYPES),
    )
)

PARAMS["extended"].update({"non_matching_tensors": _NON_MATCHING_TENSORS})

PARAMS["default"].update({"non_matching_tensors": _NON_MATCHING_TENSORS[::2]})

PARAMS["smoke"].update({"non_matching_tensors": _NON_MATCHING_TENSORS[::4]})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> RemoveRedundantInt8RequantizationPass:
    return RemoveRedundantInt8RequantizationPass()


@pytest.fixture()
def model(input_shape: Tuple[int, int, int], outputs: int) -> XCOREModel:
    model = build_fc(input_shape=input_shape, outputs=outputs)
    _glue_quantize(model.subgraphs[0].operators[0])
    return model


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(
    model: XCOREModel, trf_pass: RemoveRedundantInt8RequantizationPass
) -> None:
    subgraph = model.subgraphs[0]
    qin = subgraph.get_tensor("input")
    qout = subgraph.get_tensor("output_quantized")

    trf_pass.mutate(subgraph.operators[1])
    subgraph.sanity_check()

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.FULLY_CONNECTED
    assert len(subgraph.tensors) == 3 + 1
    assert qin in subgraph.inputs
    assert qin not in subgraph.outputs
    assert qout in subgraph.outputs
    assert qout not in subgraph.inputs


def test_non_matching_consumers(
    trf_pass: RemoveRedundantInt8RequantizationPass, model: XCOREModel
) -> None:
    _glue_quantize(model.subgraphs[0].operators[0])
    _test_non_matching_params(trf_pass, model)


def test_non_matching_op(
    trf_pass: RemoveRedundantInt8RequantizationPass, input_shape: Tuple[int, int, int]
) -> None:
    model = build_abs(input_shape=input_shape, tensor_type=TensorType.INT8)
    _glue_quantize(model.subgraphs[0].operators[0])
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
