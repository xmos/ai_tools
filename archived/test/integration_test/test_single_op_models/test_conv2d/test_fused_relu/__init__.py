# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import tensorflow as tf

from tflite2xcore.model_generation import Configuration

from .. import (
    AbstractConv2dTestModelGenerator,
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class FusedCustomReluMixin(AbstractConv2dTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config["max_value"] = cfg.pop("max_value")
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        max_value = self._config["max_value"]
        assert max_value > 0, f"max_value must be greater than 0, got {max_value}"
        assert max_value != 6, f"max_value cannot be equal to 6 (Relu6 is not custom)"

    def _build_core_model(self) -> tf.keras.Model:
        return tf.keras.Sequential(
            layers=[
                self._op_layer(input_shape=self._input_shape),
                tf.keras.layers.ReLU(max_value=self._config["max_value"]),
            ]
        )
