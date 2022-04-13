# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Optional, Tuple

from tflite2xcore.model_generation import Configuration
from tflite2xcore.model_generation.utils import parse_init_config

from . import ChannelAgnosticOpTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PReluTestModelGenerator(ChannelAgnosticOpTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config["alpha_init"] = cfg.pop("alpha_init", ("RandomUniform", -1, 1))
        super()._set_config(cfg)

    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        kwargs = {"input_shape": input_shape} if input_shape else {}
        return tf.keras.layers.PReLU(
            alpha_initializer=parse_init_config(*self._config["alpha_init"]), **kwargs
        )


GENERATOR = PReluTestModelGenerator

# TODO: fix this if/when we support prelu
CONFIGS = {"default": {0: {"height": 5, "width": 5, "channels": 3}}}


if __name__ == "__main__":
    pytest.main()
