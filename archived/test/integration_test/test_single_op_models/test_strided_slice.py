# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

import tensorflow as tf
from typing import Optional, Tuple
from math import ceil

from . import ChannelAgnosticOpTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class StridedSliceModelGenerator(ChannelAgnosticOpTestModelGenerator):
    def _build_core_model(self) -> tf.keras.Model:

        input_shape = self._input_shape

        inputs = tf.keras.Input(shape=input_shape)
        x_0 = tf.strided_slice(inputs,begin=[0, 0, 0], end= [input_shape[0], input_shape[1]//2, input_shape[2]],strides= [1, 1, 1])
        x_1 = tf.strided_slice(inputs,begin=[0, ceil(input_shape[1]/2), 0], end= [input_shape[0], input_shape[1], input_shape[2]],strides= [1, 1, 1])
        x = tf.keras.layers.Concatenate(axis=1)([x_0,x_1])
        outputs = tf.keras.layers.Flatten()(x)
        return tf.keras.Model(inputs=inputs,outputs=[outputs])
        
    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        return tf.strided_slice()

GENERATOR = StridedSliceModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main()
