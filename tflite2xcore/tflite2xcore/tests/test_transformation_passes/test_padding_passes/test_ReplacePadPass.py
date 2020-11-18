# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
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


if __name__ == "__main__":
    pytest.main()
