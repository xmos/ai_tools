# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
from abc import abstractmethod
from typing import Tuple, Optional, Type, Union

from tflite2xcore._model_generation import Configuration

from .. import (
    FilterOpTestModelGenerator,
    ChannelPreservingOpTestModelGenerator,
    test_output,
    test_converted_single_op_model,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Pool2dGenericTestModelGenerator(
    ChannelPreservingOpTestModelGenerator, FilterOpTestModelGenerator
):
    def _set_config(self, cfg: Configuration) -> None:
        strides = cfg.setdefault("strides", (2, 2))
        K_h = cfg.setdefault("K_h", 2)
        K_w = cfg.setdefault("K_w", 2)
        if cfg.setdefault("padding", "valid") == "same":
            assert (
                cfg["height"] % 2 == cfg["width"] % 2 == 0
                and K_h == K_w == 2
                and strides[0] == strides[1] == 2
            ), "same padding is only allowed for the common 2x2 case"

        super()._set_config(cfg)

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
