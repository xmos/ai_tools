# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Tuple, Optional

from tflite2xcore.model_generation.utils import parse_init_config

from . import (
    AbstractConv2dTestModelGenerator,
    ChannelPreservingOpTestModelGenerator,
)
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class DepthwiseConv2dTestModelGenerator(
    ChannelPreservingOpTestModelGenerator, AbstractConv2dTestModelGenerator
):
    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.DepthwiseConv2D:
        kwargs = {"input_shape": input_shape} if input_shape else {}
        cfg = self._config
        return tf.keras.layers.DepthwiseConv2D(
            kernel_size=(cfg["K_h"], cfg["K_w"]),
            depth_multiplier=1,
            padding=cfg["padding"],
            strides=cfg["strides"],
            bias_initializer=parse_init_config(*cfg["bias_init"]),
            kernel_initializer=parse_init_config(*cfg["weight_init"]),
            **kwargs
        )


GENERATOR = DepthwiseConv2dTestModelGenerator


if __name__ == "__main__":
    pytest.main()
