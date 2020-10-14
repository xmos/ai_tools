# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from . import build_lceBconv2d
from ..test_conv2d_passes.conftest import weight_shape  # pylint: disable=unused-import


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model():
    return build_lceBconv2d
