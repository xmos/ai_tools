# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import tensorflow as tf
from typing import Optional, Tuple

from tflite2xcore.xcore_schema import XCOREOpCodes, BuiltinOpCodes, XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.model_generation import Configuration
from tflite2xcore.model_generation.utils import parse_init_config

from . import ChannelAgnosticOpTestModelGenerator, PaddingMixin
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_reference_model_regression,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PadTestModelGenerator(ChannelAgnosticOpTestModelGenerator, PaddingMixin):
    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        return self._pad_layer(input_shape=input_shape)


GENERATOR = PadTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def abs_output_tolerance() -> int:
    return 0


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_pad


@pytest.fixture  # type: ignore
def reference_op_code() -> BuiltinOpCodes:
    return BuiltinOpCodes.PAD


if __name__ == "__main__":
    pytest.main()
