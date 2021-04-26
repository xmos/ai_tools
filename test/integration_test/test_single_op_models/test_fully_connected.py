# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Optional, Tuple

from tflite2xcore.xcore_schema import XCOREOpCodes, BuiltinOpCodes, XCOREModel
from tflite2xcore.model_generation import Configuration
from tflite2xcore.model_generation.utils import parse_init_config

from . import ChannelAgnosticOpTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class FullyConnectedTestModelGenerator(ChannelAgnosticOpTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config.update(
            {
                "weight_init": cfg.pop("weight_init", ("RandomUniform", -1, 1)),
                "bias_init": cfg.pop("bias_init", ("RandomUniform", -1, 1)),
                "outputs": cfg.pop("outputs"),
            }
        )
        super()._set_config(cfg)

    def _build_core_model(self) -> tf.keras.Model:
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Flatten(input_shape=self._input_shape),
                self._op_layer(),
            ]
        )

    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        cfg = self._config
        return tf.keras.layers.Dense(
            cfg["outputs"],
            activation="linear",
            bias_initializer=parse_init_config(*cfg["bias_init"]),
            kernel_initializer=parse_init_config(*cfg["weight_init"]),
        )


GENERATOR = FullyConnectedTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_fc


@pytest.fixture
def reference_op_code() -> BuiltinOpCodes:
    return BuiltinOpCodes.FULLY_CONNECTED


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


@pytest.mark.skip_on_device
def test_reference_model_regression(
    reference_model: XCOREModel, reference_op_code: BuiltinOpCodes
) -> None:

    operators = reference_model.subgraphs[0].operators

    assert 1 <= len(operators) <= 2
    if len(operators) == 2:
        assert operators[0].operator_code.code is BuiltinOpCodes.RESHAPE

    op_code = operators[-1].operator_code.code
    assert (
        op_code is reference_op_code
    ), f"expected: {reference_op_code}, got: {op_code}"


if __name__ == "__main__":
    pytest.main()
