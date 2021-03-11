# Copyright 2019-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import tensorflow as tf
from abc import abstractmethod
from typing import Tuple, Optional

from tflite2xcore.model_generation import Configuration
from tflite2xcore.model_generation.utils import parse_init_config

from .. import (
    PaddingMixin,
    FilterOpTestModelGenerator,
    ChannelPreservingOpTestModelGenerator,
    test_output,
    test_converted_single_op_model,
    test_idempotence,
    test_reference_model_regression,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class AbstractConv2dTestModelGenerator(FilterOpTestModelGenerator):
    @property
    def _total_width(self) -> int:
        return self._config["width"]  # type: ignore

    @property
    def _total_height(self) -> int:
        return self._config["height"]  # type: ignore

    def _set_config(self, cfg: Configuration) -> None:
        self._config.update(
            {
                "weight_init": cfg.pop("weight_init", ("RandomUniform", -1, 1)),
                "bias_init": cfg.pop("bias_init", ("Constant", 0)),
            }
        )
        cfg.setdefault("padding", "same")
        cfg.setdefault("strides", (1, 1))
        super()._set_config(cfg)


class Conv2dGenericTestModelGenerator(AbstractConv2dTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config.update(
            {
                "input_channels": cfg.pop("input_channels", 4),
                "output_channels": cfg.pop("output_channels", 4),
            }
        )
        super()._set_config(cfg)

    @property
    def _input_channels(self) -> int:
        return self._config["input_channels"]  # type: ignore

    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Conv2D:
        kwargs = {"input_shape": input_shape} if input_shape else {}
        cfg = self._config
        return tf.keras.layers.Conv2D(
            filters=cfg["output_channels"],
            kernel_size=(cfg["K_h"], cfg["K_w"]),
            padding=cfg["padding"],
            strides=cfg["strides"],
            bias_initializer=parse_init_config(*cfg["bias_init"]),
            kernel_initializer=parse_init_config(*cfg["weight_init"]),
            **kwargs,
        )


class Conv2dWordAlignedTestModelGenerator(Conv2dGenericTestModelGenerator):
    def check_config(self) -> None:
        super().check_config()
        assert (
            self._config["input_channels"] % 4 == 0
        ), "# of input channels must be multiple of 4"
        assert (
            self._config["output_channels"] % 4 == 0
        ), "# of output channels must be multiple of 4"


class Conv2dProperTestModelGenerator(Conv2dWordAlignedTestModelGenerator):
    def check_config(self) -> None:
        super().check_config()
        if self._config["padding"] == "valid":
            assert (
                self._config["K_h"] != self._total_height
                or self._config["K_w"] != self._total_width
            ), "identical kernel and image size with valid padding is reserved for single pixel testing"
