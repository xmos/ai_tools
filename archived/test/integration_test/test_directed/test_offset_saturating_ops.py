# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf

from tflite2xcore.model_generation import Configuration

from .test_mobilenet_v1 import MobileNet
from . import IntegrationTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class OffsetSaturatingModelGenerator(IntegrationTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config["layers"] = cfg.pop("layers")
        super()._set_config(cfg)

    def _build_core_model(self) -> tf.keras.Model:
        source_model = MobileNet(input_shape=(128, 128, 3), alpha=0.25)
        layers = [source_model.layers[idx] for idx in self._config["layers"]]
        input_shape = layers[0].input_shape[1:]

        return tf.keras.models.Sequential(
            layers=[tf.keras.layers.InputLayer(input_shape), *layers]
        )


GENERATOR = OffsetSaturatingModelGenerator

#  ----------------------------------------------------------------------------
#                                   CONFIGS
#  ----------------------------------------------------------------------------


CONFIGS = {
    "default": {
        0: {"layers": [1, 2, 3, 4]},
        1: {"layers": [8, 9, 10]},
        2: {"layers": [15, 16, 17]},
        3: {"layers": [21, 22, 23]},
    },
}


if __name__ == "__main__":
    pytest.main()
