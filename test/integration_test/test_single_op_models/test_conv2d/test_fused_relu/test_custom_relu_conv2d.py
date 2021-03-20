# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from tflite2xcore.xcore_schema import XCOREModel, BuiltinOpCodes

from ..test_conv2d import Conv2dTestModelGenerator
from . import FusedCustomReluMixin
from ..test_conv2d import converted_op_code  # pylint: disable=unused-import
from . import (  # pylint: disable=unused-import
    test_output,
    # test_converted_single_op_model,  # TODO: enable
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class CustomReluConv2dTestModelGenerator(
    FusedCustomReluMixin, Conv2dTestModelGenerator
):
    pass


GENERATOR = CustomReluConv2dTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def abs_output_tolerance() -> None:
    return


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


@pytest.mark.skip_on_device  # type: ignore
def test_reference_model_regression(reference_model: XCOREModel) -> None:
    operators = reference_model.subgraphs[0].operators

    opcodes = [op.operator_code.code for op in operators]
    major_version = tf.version.VERSION[:3]
    if major_version == "2.5":
        expected_opcodes = [
            BuiltinOpCodes.CONV_2D,
            BuiltinOpCodes.MINIMUM,
            BuiltinOpCodes.RELU,
        ]
    elif major_version == "2.4":
        expected_opcodes = [
            BuiltinOpCodes.CONV_2D,
            BuiltinOpCodes.QUANTIZE,
            BuiltinOpCodes.QUANTIZE,
            BuiltinOpCodes.MINIMUM,
            BuiltinOpCodes.RELU,
        ]
    else:
        expected_opcodes = [
            BuiltinOpCodes.CONV_2D,
            BuiltinOpCodes.QUANTIZE,
            BuiltinOpCodes.QUANTIZE,
            BuiltinOpCodes.MINIMUM,
            BuiltinOpCodes.QUANTIZE,
            BuiltinOpCodes.MAXIMUM,
        ]
    assert opcodes == expected_opcodes


if __name__ == "__main__":
    pytest.main()
