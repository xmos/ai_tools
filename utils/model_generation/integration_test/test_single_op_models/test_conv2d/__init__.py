# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
from abc import abstractmethod
from typing import Tuple, Optional

from tflite2xcore._model_generation import Configuration
from tflite2xcore._model_generation.utils import parse_init_config

from .. import (
    FilterOpTestModelGenerator,
    ChannelPreservingOpTestModelGenerator,
    test_output,
    test_converted_single_op_model,
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class AbstractConv2dTestModelGenerator(FilterOpTestModelGenerator):
    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        raise NotImplementedError()

    def _set_config(self, cfg: Configuration) -> None:
        self._config.update(
            dict(
                weight_init=cfg.pop("weight_init", ("RandomUniform", -1, 1)),
                bias_init=cfg.pop("bias_init", ("Constant", 0)),
            )
        )
        cfg.setdefault("padding", "same")
        cfg.setdefault("strides", (1, 1))
        super()._set_config(cfg)


class ExplicitPaddingMixin(AbstractConv2dTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        assert (
            "padding" not in cfg
        ), "padding config should be defined by (pad_t, pad_b, pad_l, pad_r)"
        cfg["padding"] = "valid"

        for side in ["t", "b", "l", "r"]:
            key = f"pad_{side}"
            self._config.update({key: cfg.pop(key, 1)})

        super()._set_config(cfg)

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


class Conv2dGenericTestModelGenerator(AbstractConv2dTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        input_channels = cfg.pop("input_channels", 4)
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        output_channels = cfg.pop("output_channels", 4)
        assert output_channels % 4 == 0, "# of output channels must be multiple of 4"

        self._config.update(
            {"input_channels": input_channels, "output_channels": output_channels}
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
