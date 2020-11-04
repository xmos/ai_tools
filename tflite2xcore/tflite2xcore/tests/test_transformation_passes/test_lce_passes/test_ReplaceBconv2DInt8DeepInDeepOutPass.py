# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest
from copy import deepcopy

from tflite2xcore.transformation_passes import ReplaceBconv2DInt8DeepInDeepOutPass
from tflite2xcore.xcore_schema import XCOREOpCodes

from . import update_lce_params
from .test_ReplaceBconv2DInt8Pass import (  # pylint: disable=unused-import
    PARAMS,
    model,
    test_matching_params,
    test_non_matching_tensors,
    test_non_matching_input_channels,
    test_non_matching_output_channels,
    test_mutate,
)

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {
        "output_channels": [16, 64, 128],
        "non_matching_output_channels": [8, 17, 28],
        "input_channels": [256, 512, 1024],
        "non_matching_input_channels": [16, 21, 128],
    }
)

PARAMS = update_lce_params(PARAMS)

# NOTE: this is intentional to reduce test counts
PARAMS["extended"] = PARAMS["default"]
PARAMS["default"] = PARAMS["smoke"]

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ReplaceBconv2DInt8DeepInDeepOutPass:
    return ReplaceBconv2DInt8DeepInDeepOutPass()


@pytest.fixture()
def new_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_bconv2d_int8_DIDO


if __name__ == "__main__":
    pytest.main()
