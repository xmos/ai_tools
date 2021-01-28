# Copyright (c) 2020, XMOS Ltd, All rights reserved

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


class SmallWeightsModelGenerator(IntegrationTestModelGenerator):
    def _build_core_model(self) -> tf.keras.Model:
        initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=2e-40)
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                4,
                4,
                kernel_initializer=initializer,
                bias_initializer=initializer,
                input_shape=(20, 20, 20),
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        return model


GENERATOR = SmallWeightsModelGenerator


#  ----------------------------------------------------------------------------
#                                   CONFIGS
#  ----------------------------------------------------------------------------

CONFIGS = {
    "default": {0: {"skip_on_device": False}},
}


if __name__ == "__main__":
    pytest.main()
