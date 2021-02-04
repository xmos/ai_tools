# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1
import pytest
from copy import deepcopy
from typing import Tuple

from tflite2xcore.transformation_passes.lce_passes import ReplaceLceQuantizePass
from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.xcore_schema import (
    TensorType,
    XCOREOpCodes,
)

from . import (
    build_LceQuantize,
    _make_name_type_pairs,
    _test_non_matching_params,
    update_lce_params,
)
from . import (  # pylint: disable=unused-import
    PARAMS,
    test_matching_params,
    test_non_matching_tensors,
    test_mutate,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {
        "non_matching_tensors": list(
            _make_name_type_pairs("input", [TensorType.FLOAT32, TensorType.INT32])
        ),
    }
)

PARAMS = update_lce_params(PARAMS)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ReplaceLceQuantizePass:
    return ReplaceLceQuantizePass()


@pytest.fixture()
def new_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_bsign_8


@pytest.fixture()
def model(input_shape: Tuple[int, int, int]) -> XCOREModel:
    return build_LceQuantize(input_shape=input_shape)


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_non_matching_input_channels(
    trf_pass: ReplaceLceQuantizePass,
    non_matching_input_channels: int,
    input_size: Tuple[int, int],
) -> None:
    model = build_LceQuantize(input_shape=(*input_size, non_matching_input_channels))
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
