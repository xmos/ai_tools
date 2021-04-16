# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Type

from tflite2xcore.xcore_schema import XCOREOpCodes, BuiltinOpCodes

from . import Pool2dGenericTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_reference_model_regression,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class MaxPool2dTestModelGenerator(Pool2dGenericTestModelGenerator):
    @property
    def _op_class(self) -> Type[tf.keras.layers.MaxPool2D]:
        return tf.keras.layers.MaxPool2D  # type: ignore


GENERATOR = MaxPool2dTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_maxpool2d


@pytest.fixture
def reference_op_code() -> BuiltinOpCodes:
    return BuiltinOpCodes.MAX_POOL_2D


if __name__ == "__main__":
    pytest.main()
