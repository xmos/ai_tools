# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from tflite2xcore.xcore_schema import XCOREOpCodes


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_lookup_8


@pytest.fixture
def abs_output_tolerance() -> int:
    return 0
