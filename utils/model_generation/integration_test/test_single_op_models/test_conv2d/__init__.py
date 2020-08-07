# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore

from tflite2xcore._model_generation import Configuration
from tflite2xcore._model_generation.utils import parse_init_config

from .. import IntegrationTestModelGenerator, test_output


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Conv2dGenericTestModelGenerator(IntegrationTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        input_channels = cfg.pop("input_channels", 4)
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        output_channels = cfg.pop("output_channels", 4)
        assert output_channels % 4 == 0, "# of output channels must be multiple of 4"

        self._config = dict(
            K_w=cfg.pop("K_w", 3),
            K_h=cfg.pop("K_h", 3),
            height=cfg.pop("height", 5),
            width=cfg.pop("width", 5),
            input_channels=input_channels,
            output_channels=output_channels,
            padding=cfg.pop("padding", "same"),
            strides=cfg.pop("strides", (1, 1)),
            weight_init=cfg.pop("weight_init", ("RandomUniform", -1, 1)),
            bias_init=cfg.pop("bias_init", ("Constant", 0)),
        )
        super()._set_config(cfg)

    def build_core_model(self) -> tf.keras.Model:
        cfg = self._config
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(
                    filters=cfg["output_channels"],
                    kernel_size=(cfg["K_h"], cfg["K_w"]),
                    padding=cfg["padding"],
                    strides=cfg["strides"],
                    input_shape=(cfg["height"], cfg["width"], cfg["input_channels"]),
                    bias_initializer=parse_init_config(*cfg["bias_init"]),
                    kernel_initializer=parse_init_config(*cfg["weight_init"]),
                )
            ],
        )

    def build(self) -> None:
        self._prep_backend()
        try:
            self._model = self.build_core_model()
        except ValueError as e:
            if e.args[0].startswith("Negative dimension size caused by"):
                raise ValueError(
                    "Negative dimension size (Hint: if using 'valid' padding "
                    "verify that the kernel is at least the size of input image)"
                ) from e
            else:
                raise
        self._model.build()
