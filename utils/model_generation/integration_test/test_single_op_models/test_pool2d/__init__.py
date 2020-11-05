# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf
from abc import abstractmethod
from typing import Tuple, Optional, Type, Union

from tflite2xcore.model_generation import Configuration

from .. import (
    FilterOpTestModelGenerator,
    ChannelPreservingOpTestModelGenerator,
    test_output,
    test_converted_single_op_model,
    test_reference_model_regression,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Pool2dGenericTestModelGenerator(
    ChannelPreservingOpTestModelGenerator, FilterOpTestModelGenerator
):
    def _set_config(self, cfg: Configuration) -> None:
        cfg.setdefault("strides", (2, 2))
        cfg.setdefault("K_h", 2)
        cfg.setdefault("K_w", 2)
        cfg.setdefault("padding", "valid")
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        if self._config["padding"] == "same":
            assert (
                self._config["height"] % 2 == self._config["width"] % 2 == 0
                and self._config["K_h"] == self._config["K_w"] == 2
                and self._config["strides"][0] == self._config["strides"][1] == 2
            ), "same padding is only allowed for the common 2x2 case"

    @property
    @abstractmethod
    def _op_class(
        self,
    ) -> Union[Type[tf.keras.layers.MaxPool2D], Type[tf.keras.layers.AvgPool2D]]:
        raise NotImplementedError()

    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        kwargs = {"input_shape": input_shape} if input_shape else {}
        cfg = self._config
        return self._op_class(
            pool_size=(cfg["K_h"], cfg["K_w"]),
            strides=cfg["strides"],
            padding=cfg["padding"],
            **kwargs
        )
