# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from ..test_depthwise_conv2d import DepthwiseConv2dTestModelGenerator
from . import ExplicitPaddingMixin
from ..test_depthwise_conv2d import converted_op_code  # pylint: disable=unused-import
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PaddedDepthwiseConv2dTestModelGenerator(
    ExplicitPaddingMixin, DepthwiseConv2dTestModelGenerator
):
    pass


GENERATOR = PaddedDepthwiseConv2dTestModelGenerator


if __name__ == "__main__":
    pytest.main()
