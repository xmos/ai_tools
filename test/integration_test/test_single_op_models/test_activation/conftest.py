# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_lookup_8


@pytest.fixture  # type: ignore
def abs_output_tolerance() -> int:
    return 0
