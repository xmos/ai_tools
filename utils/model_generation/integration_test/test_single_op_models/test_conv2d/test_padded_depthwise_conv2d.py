# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import tensorflow as tf  # type: ignore

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this

from . import ExplicitPaddingMixin
from .test_depthwise_conv2d import DepthwiseConv2dTestModelGenerator, test_output


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PaddedDepthwiseConv2dTestModelGenerator(
    ExplicitPaddingMixin, DepthwiseConv2dTestModelGenerator
):
    pass


GENERATOR = PaddedDepthwiseConv2dTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_converted_model(xcore_model: XCOREModel) -> None:
    operators = xcore_model.subgraphs[0].operators
    assert len(operators) == 1
    op = operators[0]
    assert op.operator_code.code is XCOREOpCodes.XC_conv2d_depthwise


if __name__ == "__main__":
    pytest.main()
