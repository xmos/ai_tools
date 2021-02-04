# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest
import tensorflow as tf
from typing import Type

from tflite2xcore.xcore_schema import XCOREOpCodes, BuiltinOpCodes  # type: ignore # TODO: fix this

from . import Pool2dGenericTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_reference_model_regression,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class AvgPool2dTestModelGenerator(Pool2dGenericTestModelGenerator):
    @property
    def _op_class(self) -> Type[tf.keras.layers.AvgPool2D]:
        return tf.keras.layers.AvgPool2D  # type: ignore


GENERATOR = AvgPool2dTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_avgpool2d


@pytest.fixture  # type: ignore
def reference_op_code() -> BuiltinOpCodes:
    return BuiltinOpCodes.AVERAGE_POOL_2D


if __name__ == "__main__":
    pytest.main()
