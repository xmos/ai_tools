# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1
import pytest
from copy import deepcopy

from tflite2xcore.transformation_passes import ReplaceBconv2DBitpackedDeepInPass
from tflite2xcore.xcore_schema import XCOREOpCodes

from .test_ReplaceBconv2DBitpackedPass import (  # pylint: disable=unused-import
    PARAMS,
    model,
    test_matching_params,
    test_non_matching_tensors,
    test_non_matching_input_channels,
    test_mutate,
)
from .test_ReplaceBconv2DInt8DeepInDeepOutPass import PARAMS as DEEP_PARAMS

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------


PARAMS = deepcopy(PARAMS)

for key in PARAMS:
    PARAMS[key]["input_channels"] = DEEP_PARAMS[key]["input_channels"]
    PARAMS[key]["non_matching_input_channels"] = DEEP_PARAMS[key][
        "non_matching_input_channels"
    ]

# NOTE: this is intentional to reduce test counts
PARAMS["extended"] = PARAMS["default"]
PARAMS["default"] = PARAMS["smoke"]


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ReplaceBconv2DBitpackedDeepInPass:
    return ReplaceBconv2DBitpackedDeepInPass()


@pytest.fixture()
def new_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_bconv2d_bin_DI


if __name__ == "__main__":
    pytest.main()
