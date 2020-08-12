# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import tensorflow as tf  # type: ignore

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore._model_generation.utils import parse_init_config
from tflite2xcore._model_generation import Configuration

from . import AbstractConv2dTestModelGenerator, IntegrationTestRunner, test_output


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class DepthwiseConv2dTestModelGenerator(AbstractConv2dTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        channels = cfg.pop("channels", 4)
        assert channels % 4 == 0, "# of channels must be multiple of 4"
        self._config["channels"] = channels

        super()._set_config(cfg)

    def build_core_model(self) -> tf.keras.Model:
        cfg = self._config
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(cfg["K_h"], cfg["K_w"]),
                    depth_multiplier=1,
                    padding=cfg["padding"],
                    strides=cfg["strides"],
                    input_shape=(cfg["height"], cfg["width"], cfg["channels"]),
                    bias_initializer=parse_init_config(*cfg["bias_init"]),
                    kernel_initializer=parse_init_config(*cfg["weight_init"]),
                )
            ],
        )


GENERATOR = DepthwiseConv2dTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_converted_model(xcore_model: XCOREModel) -> None:
    operators = xcore_model.subgraphs[0].operators
    assert len(operators) == 1
    op = operators[0]
    assert op.operator_code.code is XCOREOpCodes.XC_conv2d_depthwise


if __name__ == "__main__":
    pytest.main()
