# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Callable

from tflite2xcore.xcore_schema import BuiltinOpCodes

from . import LUTActivationOpTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_reference_model_regression,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Relu6TestModelGenerator(LUTActivationOpTestModelGenerator):
    @property
    def act_fun(self) -> Callable[[tf.Tensor], tf.Tensor]:
        return lambda x: tf.nn.relu6(x)


GENERATOR = Relu6TestModelGenerator

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture
def reference_op_code() -> BuiltinOpCodes:
    return BuiltinOpCodes.RELU6


if __name__ == "__main__":
    pytest.main()
