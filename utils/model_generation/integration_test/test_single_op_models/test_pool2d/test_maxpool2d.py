# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import tensorflow as tf  # type: ignore
from typing import Type

from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this

from . import (
    Pool2dGenericTestModelGenerator,
    test_output,
    test_converted_single_op_model,
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


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_maxpool2d


if __name__ == "__main__":
    pytest.main()
