# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Type
from tensorflow.keras import layers,models

from tflite2xcore.xcore_schema import XCOREOpCodes, BuiltinOpCodes

from . import (  # pylint: disable=unused-import
    test_output_tdnn,
)
from .. import IntegrationTestModelGenerator, TdnnTestRunner


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class TutorialsImagesCnnModelGenerator(IntegrationTestModelGenerator):
    def _build_core_model(self) -> tf.keras.Model:
        model = models.Sequential()
        # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.Input(shape=(32,32,3)))
        model.add(layers.MaxPooling2D((2, 2),strides=(1,1)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2),strides=(1,1)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))  
        # model.add(layers.Flatten())
        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dense(10))
        return model

GENERATOR = TutorialsImagesCnnModelGenerator


#  ----------------------------------------------------------------------------


RUNNER = TdnnTestRunner


if __name__ == "__main__":
    pytest.main()
