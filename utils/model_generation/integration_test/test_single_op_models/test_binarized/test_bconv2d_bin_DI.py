# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import larq
import tensorflow as tf

from tflite2xcore.xcore_schema import ExternalOpCodes, XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore.model_generation import Configuration

from . import BConv2dGenericTestModelGenerator

from .test_bconv2d_bin import BConv2dBitpackedTestRunner

from . import (  # pylint: disable=unused-import
    test_reference_model_regression,
    test_converted_single_op_model,
    test_mean_abs_diffs,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class BConv2dBitpackedDeepInTestModelGenerator(BConv2dGenericTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        cfg.setdefault("padding", "valid")
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        assert (
            self._config["input_channels"] % 256 == 0
        ), "# of input channels must be multiple of 256"

    def _build_core_model(self) -> tf.keras.Model:
        img = tf.keras.layers.Input(shape=self._input_shape)
        x = self._fake_quant(img)
        x = self._op_layer()(x)
        # NOTE: we need the next dummy layer to produce a bconv2d with bitpacked output
        x = larq.layers.QuantConv2D(
            filters=32,
            kernel_size=(1, 1),
            padding="valid",
            pad_values=1,
            strides=(1, 1),
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
        )(x)
        x = self._fake_quant(x)
        return tf.keras.Model(img, x)


GENERATOR = BConv2dBitpackedDeepInTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


RUNNER = BConv2dBitpackedTestRunner

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
