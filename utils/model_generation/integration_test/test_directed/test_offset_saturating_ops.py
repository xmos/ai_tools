# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import tensorflow as tf  # type: ignore

from tflite2xcore._model_generation import Configuration

from . import IntegrationTestModelGenerator, test_output, test_idempotence


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class OffsetSaturatingModel(IntegrationTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config.update({"layers": cfg.pop("layers")})
        super()._set_config(cfg)

    def _build_core_model(self) -> tf.keras.Model:
        source_model = tf.keras.applications.MobileNet(
            input_shape=(128, 128, 3), alpha=0.25
        )
        layers = [source_model.layers[idx] for idx in self._config["layers"]]
        import logging

        logging.warning(layers)
        input_shape = layers[0].input_shape[1:]

        return tf.keras.models.Sequential(
            layers=[tf.keras.layers.InputLayer(input_shape), *layers]
        )


GENERATOR = OffsetSaturatingModel

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
