# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf

from tflite2xcore.utils import asserting_cast
from tflite2xcore.xcore_schema import XCOREModel, ValidOpCodes, BuiltinOpCodes
from tflite2xcore.model_generation import Configuration

from .. import (
    PaddingMixin,
    AbstractConv2dTestModelGenerator,
    test_output,
    test_converted_single_op_model,
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class ExplicitlyPaddedConv2dMixin(PaddingMixin, AbstractConv2dTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        assert (
            "padding" not in cfg
        ), f"padding config should be defined by {self._PAD_KEYS}"
        cfg["padding"] = "valid"

        super()._set_config(cfg)

    @property
    def _total_width(self) -> int:
        return (
            super()._total_width
            + asserting_cast(int, self._config["pad_l"])
            + asserting_cast(int, self._config["pad_r"])
        )

    @property
    def _total_height(self) -> int:
        return (
            super()._total_height
            + asserting_cast(int, self._config["pad_t"])
            + asserting_cast(int, self._config["pad_b"])
        )

    def _build_core_model(self) -> tf.keras.Model:

        return tf.keras.Sequential(
            layers=[self._pad_layer(input_shape=self._input_shape), self._op_layer()]
        )


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


@pytest.mark.skip_on_device
def test_reference_model_regression(
    reference_model: XCOREModel, reference_op_code: ValidOpCodes
) -> None:
    operators = reference_model.subgraphs[0].operators
    assert len(operators) == 2
    assert operators[0].operator_code.code is BuiltinOpCodes.PAD

    op_code = operators[1].operator_code.code
    assert (
        op_code is reference_op_code
    ), f"expected: {reference_op_code}, got: {op_code}"
