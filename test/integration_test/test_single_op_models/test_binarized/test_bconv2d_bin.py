# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import larq
import tensorflow as tf

from tflite2xcore.xcore_schema import ExternalOpCodes, XCOREOpCodes
from tflite2xcore.model_generation import Configuration

from . import (
    BinarizedSingleOpRunner,
    BConv2dGenericTestModelGenerator,
    LarqSingleOpConverter,
)

from . import (  # pylint: disable=unused-import
    test_reference_model_regression,
    test_converted_single_op_model,
    test_mean_abs_diffs,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class BConv2dBitpackedTestModelGenerator(BConv2dGenericTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        cfg.setdefault("padding", "valid")
        for forbidden_key in ("activation", "output_range"):
            assert (
                forbidden_key not in cfg
            ), f"{forbidden_key} cannot be specified for BConv2dBitpacked tests"
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        assert (
            self._config["input_channels"] % 32 == 0
        ), "# of input channels must be multiple of 32"

    def _build_core_model(self) -> tf.keras.Model:
        img = tf.keras.layers.Input(shape=self._input_shape)
        x = self._fake_quant(img, *self._config["input_range"])
        x = self._op_layer()(x)
        # NOTE: we need the next dummy layer in order to produce a bconv2d with bitpacked output
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
        x = self._fake_quant(x, *self._config["output_range"])
        return tf.keras.Model(img, x)


GENERATOR = BConv2dBitpackedTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


class BConv2dBitpackedTestRunner(BinarizedSingleOpRunner):
    def make_lce_converter(self) -> LarqSingleOpConverter:
        return LarqSingleOpConverter(
            self, self.get_built_model, strip=True, remove_last_op=True
        )


RUNNER = BConv2dBitpackedTestRunner


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
