# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import BuiltinOpCodes  # type: ignore # TODO: fix this

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def reference_op_code() -> BuiltinOpCodes:
    return BuiltinOpCodes.CONV_2D
