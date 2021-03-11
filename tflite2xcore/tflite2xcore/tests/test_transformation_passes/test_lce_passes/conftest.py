# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from . import build_lceBconv2d
from ..test_conv2d_passes.conftest import weight_shape  # pylint: disable=unused-import


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model():
    return build_lceBconv2d

