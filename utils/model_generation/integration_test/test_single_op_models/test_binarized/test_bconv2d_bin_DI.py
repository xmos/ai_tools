# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import tensorflow as tf

from tflite2xcore.xcore_schema import ExternalOpCodes, XCOREOpCodes  # type: ignore # TODO: fix this

from . import (
    BinarizedTestRunner,
    BConv2dGenericTestModelGenerator,
    LarqConverter,
)

from .test_bconv2d_bin import BConv2dBitpackedTestRunner

from . import (
    test_reference_model_regression,
    # test_converted_single_op_model,  # TODO: enable this
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class BConv2dBitpackedDeepInTestModelGenerator(BConv2dGenericTestModelGenerator):
    def check_config(self) -> None:
        super().check_config()
        assert (
            self._config["input_channels"] % 256 == 0
        ), "# of input channels must be multiple of 256"

    def _build_core_model(self) -> tf.keras.Model:
        img = tf.keras.layers.Input(shape=self._input_shape)
        x = self._fake_quant(img)
        x = self._op_layer()(x)
        x = self._op_layer()(x)
        x = self._fake_quant(x)
        return tf.keras.Model(img, x)


GENERATOR = BConv2dBitpackedDeepInTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


RUNNER = BConv2dBitpackedTestRunner

#  ----------------------------------------------------------------------------
#                                   CONFIGS
#  ----------------------------------------------------------------------------

CONFIGS = {  # TODO: generate random configs
    "default": {
        0: {
            "input_channels": 256,
            "output_channels": 64,
            "K_h": 3,
            "K_w": 3,
            "height": 8,
            "width": 8,
        },
    },
}

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def reference_op_code() -> ExternalOpCodes:
    return ExternalOpCodes.LceBconv2d


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_bconv2d_bin_DI


if __name__ == "__main__":
    pytest.main()
