# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from ..test_depthwise_conv2d import DepthwiseConv2dTestModelGenerator
from . import ExplicitlyPaddedConv2dMixin
from ..test_depthwise_conv2d import (  # pylint: disable=unused-import
    reference_op_code,
    converted_op_code,
)
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_idempotence,
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
