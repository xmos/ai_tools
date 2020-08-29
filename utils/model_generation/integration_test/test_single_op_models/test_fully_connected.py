# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import tensorflow as tf  # type: ignore
from typing import Optional, Tuple

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore._model_generation import Configuration
from tflite2xcore._model_generation.utils import parse_init_config

from . import ChannelAgnosticOpTestModelGenerator, test_output


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class FullyConnectedTestModelGenerator(ChannelAgnosticOpTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config.update(
            dict(
                weight_init=cfg.pop("weight_init", ("RandomUniform", -1, 1)),
                bias_init=cfg.pop("bias_init", ("RandomUniform", -1, 1)),
                outputs=cfg.pop("outputs"),
            )
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
#                                   TESTS
#  ----------------------------------------------------------------------------

# TODO: fix this when fully connected is changed to output int8
def test_converted_single_op_model(xcore_model: XCOREModel) -> None:
    operators = xcore_model.subgraphs[0].operators
    assert len(operators) == 2
    assert operators[0].operator_code.code is XCOREOpCodes.XC_fc_deepin_anyout
    assert operators[1].operator_code.code is XCOREOpCodes.XC_requantize_16_to_8


if __name__ == "__main__":
    pytest.main()
