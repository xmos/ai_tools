# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import tensorflow as tf  # type: ignore
from typing import Tuple, Optional

from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore._model_generation.utils import parse_init_config
from tflite2xcore._model_generation import Configuration

from . import (
    AbstractConv2dTestModelGenerator,
    test_output,
    test_converted_single_op_model,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class DepthwiseConv2dTestModelGenerator(AbstractConv2dTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        channels = cfg.pop("channels", 4)
        assert channels % 4 == 0, "# of channels must be multiple of 4"
        self._config.update({"channels": channels})

        super()._set_config(cfg)

    @property
    def _input_channels(self) -> int:
        return self._config["channels"]

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


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_conv2d_depthwise


if __name__ == "__main__":
    pytest.main()
