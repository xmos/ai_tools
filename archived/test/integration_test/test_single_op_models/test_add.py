# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

import tensorflow as tf
from typing import Optional, Tuple
import random

from . import ChannelAgnosticOpTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class AddModelGenerator(ChannelAgnosticOpTestModelGenerator):
    def _build_core_model(self) -> tf.keras.Model:
        input = tf.keras.Input(shape=self._input_shape)
        x2 = tf.random.normal([1, *self._input_shape], mean=random.random())
        out = self._op_layer()([input, x2])
        return tf.keras.models.Model(inputs=input, outputs=out)

    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        return tf.keras.layers.Add()


GENERATOR = AddModelGenerator


if __name__ == "__main__":
    pytest.main()
