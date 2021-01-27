# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import logging
import tensorflow as tf
from pathlib import Path
from tflite2xcore.utils import LoggingContext  # type: ignore # TODO: fix this

from . import IntegrationTestModelGenerator

from . import test_idempotence  # pylint: disable=unused-import


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class SmallWeightsModelGenerator(IntegrationTestModelGenerator):
    def _build_core_model(self) -> tf.keras.Model:
        # tf may complain about missing gradients, so silence it
        with LoggingContext(tf.get_logger(), logging.ERROR):
            return tf.keras.models.load_model(
                Path(__file__).parent / "small_weights_model", compile=False
            )


GENERATOR = SmallWeightsModelGenerator


#  ----------------------------------------------------------------------------
#                                   CONFIGS
#  ----------------------------------------------------------------------------

CONFIGS = {
    "default": {0: {"skip_on_device": False}},
}


if __name__ == "__main__":
    pytest.main()
