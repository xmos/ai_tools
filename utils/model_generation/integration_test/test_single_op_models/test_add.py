# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest


import tensorflow as tf
from typing import Optional, Tuple

from tflite2xcore.xcore_schema import XCOREOpCodes, BuiltinOpCodes, XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.model_generation import Configuration
from tflite2xcore.model_generation.utils import parse_init_config

from . import ChannelPreservingOpTestModelGenerator
from . import (  # pylint: disable=unused-import
    # test_output,
    test_converted_single_op_model,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class AddModelGenerator(ChannelPreservingOpTestModelGenerator):
    def _build_core_model(self) -> tf.keras.Model:
        input = tf.keras.Input(shape=self._input_shape)
        return tf.keras.models.Model(
            inputs=input, outputs=self._op_layer()([input, input])
        )

    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        return tf.keras.layers.Add()


GENERATOR = AddModelGenerator

#  ----------------------------------------------------------------------------
#                                   CONFIGS
#  ----------------------------------------------------------------------------


CONFIGS = {
    "default": {0: {"height": 4, "width": 4}},
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_add_8


if __name__ == "__main__":
    pytest.main()
