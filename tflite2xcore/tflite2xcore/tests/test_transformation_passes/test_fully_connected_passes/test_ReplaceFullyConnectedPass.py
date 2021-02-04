# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from tflite2xcore.xcore_schema import XCOREOpCodes
from tflite2xcore.transformation_passes import ReplaceFullyConnectedPass

from .conftest import (
    PARAMS,
    test_matching_params,
    test_non_matching_tensors,
    test_replace_mutate as test_mutate,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ReplaceFullyConnectedPass:
    return ReplaceFullyConnectedPass()


@pytest.fixture()
def new_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_fc


if __name__ == "__main__":
    pytest.main()
