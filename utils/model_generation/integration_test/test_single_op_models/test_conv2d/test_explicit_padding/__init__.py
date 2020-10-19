# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf

from tflite2xcore.model_generation import Configuration

from .. import (
    AbstractConv2dTestModelGenerator,
    test_output,
    test_converted_single_op_model,
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class ExplicitPaddingMixin(AbstractConv2dTestModelGenerator):
    _PAD_KEYS = ("pad_t", "pad_b", "pad_l", "pad_r")

    def _set_config(self, cfg: Configuration) -> None:
        assert (
            "padding" not in cfg
        ), f"padding config should be defined by {self._PAD_KEYS}"
        cfg["padding"] = "valid"

        self._config.update({key: cfg.pop(key, 1) for key in self._PAD_KEYS})
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        for key in self._PAD_KEYS:
            assert self._config[key] >= 0, f"{key} must non-negative"

    @property
    def _total_width(self) -> int:
        return super()._total_width + self._config["pad_l"] + self._config["pad_r"]  # type: ignore

    @property
    def _total_height(self) -> int:
        return super()._total_height + self._config["pad_t"] + self._config["pad_b"]  # type: ignore

    def _build_core_model(self) -> tf.keras.Model:
        cfg = self._config
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.ZeroPadding2D(
                    padding=(
                        (cfg["pad_t"], cfg["pad_b"]),
                        (cfg["pad_l"], cfg["pad_r"]),
                    ),
                    input_shape=self._input_shape,
                ),
                self._op_layer(),
            ]
        )

