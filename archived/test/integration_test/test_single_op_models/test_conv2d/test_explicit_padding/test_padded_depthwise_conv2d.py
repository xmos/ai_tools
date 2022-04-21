# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from ..test_depthwise_conv2d import DepthwiseConv2dTestModelGenerator
from . import ExplicitlyPaddedConv2dMixin
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PaddedDepthwiseConv2dTestModelGenerator(
    ExplicitlyPaddedConv2dMixin, DepthwiseConv2dTestModelGenerator
):
    pass


GENERATOR = PaddedDepthwiseConv2dTestModelGenerator


if __name__ == "__main__":
    pytest.main()
