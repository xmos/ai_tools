# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import numpy as np
from typing import Tuple
from copy import deepcopy

from tflite2xcore.transformation_passes import ReplacePadPass
from tflite2xcore.xcore_schema import BuiltinOpCodes, XCOREModel, XCOREOpCodes

from . import PaddingType
from ..model_builders import build_pad, ModelBuilder

from . import test_non_matching_paddings, test_replace_mutate as _test_replace_mutate
from .conftest import (
    PARAMS,
    test_matching_params,
    update_params_with_paddings,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------


def is_matching(padding: PaddingType) -> bool:
    return padding[0] == padding[3] == (0, 0)


PARAMS = update_params_with_paddings(deepcopy(PARAMS), is_matching=is_matching)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()  # type: ignore
def build_model() -> ModelBuilder:
    return build_pad


@pytest.fixture()  # type: ignore
def trf_pass() -> ReplacePadPass:
    return ReplacePadPass()


@pytest.fixture()
def new_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_pad


@pytest.fixture()  # type: ignore
def model(input_shape: Tuple[int, int, int, int], paddings: PaddingType) -> XCOREModel:
    return build_pad(input_shape=input_shape, paddings=paddings)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(
    trf_pass: ReplacePadPass, model: XCOREModel, new_opcode: XCOREOpCodes
) -> None:
    # extract original padding values
    subgraph = model.subgraphs[0]
    params_ori = subgraph.operators[-1].inputs[1].as_array().tolist()

    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    _test_replace_mutate(trf_pass, model, new_opcode)

    # check operators
    operators = subgraph.operators
    assert len(operators) == 1
    op = operators[0]

    # check tensors
    assert len(op.inputs) == 2
    assert len(op.outputs) == 1

    # check parameters
    params_new = op.inputs[1].as_array().tolist()
    assert params_new == params_ori

    zero_point_byte = np.int8(op.inputs[0].quantization["zero_point"][0]).tobytes()
    pad_value_bytes = np.int32(op.custom_options["pad_value"]).tobytes()
    assert pad_value_bytes == zero_point_byte * 4


if __name__ == "__main__":
    pytest.main()
