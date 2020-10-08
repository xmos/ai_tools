# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import tensorflow as tf  # type: ignore
from typing import Optional, Tuple

from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this

from . import ChannelPreservingOpTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class GlobalAveragePooling2dTestModelGenerator(ChannelPreservingOpTestModelGenerator):
    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        kwargs = {"input_shape": input_shape} if input_shape else {}
        return tf.keras.layers.GlobalAveragePooling2D(**kwargs)


GENERATOR = GlobalAveragePooling2dTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_avgpool2d_global


if __name__ == "__main__":
    pytest.main()
