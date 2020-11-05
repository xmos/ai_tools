# Copyright (c) 2020, XMOS Ltd, All rights reserved

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
