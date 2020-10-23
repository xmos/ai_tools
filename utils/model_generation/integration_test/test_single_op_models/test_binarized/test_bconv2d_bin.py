# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import tensorflow as tf

pytestmark = pytest.mark.skip  # TODO: remove this

from tflite2xcore.xcore_schema import ExternalOpCodes, XCOREOpCodes  # type: ignore # TODO: fix this

from . import (
    BinarizedTestRunner,
    BConv2dGenericTestModelGenerator,
    LarqConverter,
)

from . import (  # pylint: disable=unused-import
    test_reference_model_regression,
    # test_converted_single_op_model,  # TODO: enable this
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class BConv2dBitpackedTestModelGenerator(BConv2dGenericTestModelGenerator):
    def _build_core_model(self) -> tf.keras.Model:
        img = tf.keras.layers.Input(shape=self._input_shape)
        x = self._fake_quant(img)
        x = self._op_layer()(x)
        x = self._op_layer()(x)
        x = self._fake_quant(x)
        return tf.keras.Model(img, x)


GENERATOR = BConv2dBitpackedTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


class BConv2dBitpackedTestRunner(BinarizedTestRunner):
    def make_lce_converter(self) -> LarqConverter:
        return LarqConverter(
            self, self.get_built_model, strip=True, remove_last_op=True
        )


RUNNER = BConv2dBitpackedTestRunner

#  ----------------------------------------------------------------------------
#                                   CONFIGS
#  ----------------------------------------------------------------------------

CONFIGS = {  # TODO: generate random configs
    "default": {
        0: {
            "input_channels": 32,
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
    return XCOREOpCodes.XC_bconv2d_bin


if __name__ == "__main__":
    pytest.main()
