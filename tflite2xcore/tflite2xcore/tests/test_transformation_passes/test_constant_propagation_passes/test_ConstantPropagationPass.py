# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
from typing import Tuple

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import ConstantPropagationPass
from tflite2xcore.converter import CleanupManager

from . import build_quantize_dequantize_identity

from . import (  # pylint: disable=unused-import
    PARAMS,
    test_matching_params as _test_matching_params,
    _test_non_matching_params,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ConstantPropagationPass:
    return ConstantPropagationPass()


@pytest.fixture()
def model(input_shape: Tuple[int, int, int]) -> XCOREModel:
    return build_quantize_dequantize_identity(input_shape=input_shape)


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_matching_params(trf_pass: ConstantPropagationPass, model: XCOREModel) -> None:
    _test_matching_params(trf_pass, model, op_idx=0)


def test_non_matching_input_tensor(
    trf_pass: ConstantPropagationPass, model: XCOREModel
) -> None:
    _test_non_matching_params(trf_pass, model, op_idx=1)


def test_mutate(trf_pass: ConstantPropagationPass, model: XCOREModel) -> None:
    subgraph = model.subgraphs[0]
    operators = subgraph.operators
    old_output = subgraph.outputs[0]
    assert not old_output.buffer.data

    data = operators[0].inputs[0].buffer.data

    # run mutating pass, then clean up
    trf_pass.run(model)
    model.sanity_check()
    CleanupManager(model).run_passes()

    assert not operators
    assert len(subgraph.tensors) == 1
    assert subgraph.tensors[0] is old_output
    assert old_output.buffer.data == data


if __name__ == "__main__":
    pytest.main()
