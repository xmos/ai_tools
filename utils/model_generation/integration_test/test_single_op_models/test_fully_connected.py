# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import tensorflow as tf  # type: ignore
from typing import Optional, Tuple

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore._model_generation import Configuration
from tflite2xcore._model_generation.utils import parse_init_config

from . import (
    ChannelAgnosticOpTestModelGenerator,
    test_output,
    test_idempotence,
    test_converted_single_op_model,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class FullyConnectedTestModelGenerator(ChannelAgnosticOpTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config.update(
            {
                "weight_init": cfg.pop("weight_init", ("RandomUniform", -1, 1)),
                "bias_init": cfg.pop("bias_init", ("RandomUniform", -1, 1)),
                "outputs": cfg.pop("outputs"),
            }
        )
        super()._set_config(cfg)

    def _build_core_model(self) -> tf.keras.Model:
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Flatten(input_shape=self._input_shape),
                self._op_layer(),
            ]
        )

    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        cfg = self._config
        return tf.keras.layers.Dense(
            cfg["outputs"],
            activation="linear",
            bias_initializer=parse_init_config(*cfg["bias_init"]),
            kernel_initializer=parse_init_config(*cfg["weight_init"]),
        )


GENERATOR = FullyConnectedTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_fc


if __name__ == "__main__":
    pytest.main()
