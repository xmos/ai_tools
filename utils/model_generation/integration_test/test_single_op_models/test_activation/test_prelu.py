# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import tensorflow as tf  # type: ignore
from typing import Optional, Tuple

from tflite2xcore.xcore_schema import BuiltinOpCodes  # type: ignore # TODO: fix this
from tflite2xcore._model_generation import Configuration
from tflite2xcore._model_generation.utils import parse_init_config

from . import (
    ChannelAgnosticOpTestModelGenerator,
    test_output,
    test_converted_single_op_model,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PReluTestModelGenerator(ChannelAgnosticOpTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config.update(
            {"alpha_init": cfg.pop("alpha_init", ("RandomUniform", -1, 1))}
        )
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


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def converted_op_code() -> BuiltinOpCodes:
    return BuiltinOpCodes.PRELU  # TODO: fix this if/when we support prelu


if __name__ == "__main__":
    pytest.main()
