# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest
import tensorflow as tf

from . import IntegrationTestModelGenerator

from . import (  # pylint: disable=unused-import
    test_idempotence,
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class ZeroWeightsModelGenerator(IntegrationTestModelGenerator):
    def _build_core_model(self) -> tf.keras.Model:
        initializer = tf.keras.initializers.Constant(0)
        return tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    4,
                    4,
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                    input_shape=(20, 20, 20),
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )


GENERATOR = ZeroWeightsModelGenerator


if __name__ == "__main__":
    pytest.main()
