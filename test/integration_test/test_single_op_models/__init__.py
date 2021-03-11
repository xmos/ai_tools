# Copyright 2019-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from abc import abstractmethod
from typing import Tuple, Optional

from tflite2xcore.model_generation import Configuration
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import XCOREOpCodes, ValidOpCodes

from .. import (
    IntegrationTestRunner,
    BinarizedTestRunner,
    _compare_batched_arrays,
    BatchedArrayComparison,
    IntegrationTestModelGenerator,
    test_output,
    test_mean_abs_diffs,
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class ImageInputOpTestModelGenerator(IntegrationTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config.update({key: cfg.pop(key) for key in ["height", "width"]})
        super()._set_config(cfg)

    @property
    @abstractmethod
    def _input_channels(self) -> int:
        raise NotImplementedError()

    @property
    def _input_shape(self) -> Tuple[int, int, int]:
        cfg = self._config
        return cfg["height"], cfg["width"], self._input_channels

    @abstractmethod
    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        raise NotImplementedError()

    def _build_core_model(self) -> tf.keras.Model:
        return tf.keras.Sequential(
            layers=[self._op_layer(input_shape=self._input_shape)]
        )


class PaddingMixin(ImageInputOpTestModelGenerator):
    _PAD_KEYS = ("pad_t", "pad_b", "pad_l", "pad_r")

    def _set_config(self, cfg: Configuration) -> None:
        self._config.update({key: cfg.pop(key, 1) for key in self._PAD_KEYS})
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        for key in self._PAD_KEYS:
            assert self._config[key] >= 0, f"{key} must be non-negative"

    def _pad_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        kwargs = {"input_shape": input_shape} if input_shape else {}
        cfg = self._config
        return tf.keras.layers.ZeroPadding2D(
            padding=((cfg["pad_t"], cfg["pad_b"]), (cfg["pad_l"], cfg["pad_r"])),
            **kwargs,
        )


class ChannelAgnosticOpTestModelGenerator(ImageInputOpTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config["channels"] = cfg.pop("channels", 4)
        super()._set_config(cfg)

    @property
    def _input_channels(self) -> int:
        return self._config["channels"]  # type: ignore


class ChannelPreservingOpTestModelGenerator(ChannelAgnosticOpTestModelGenerator):
    def check_config(self) -> None:
        super().check_config()
        assert self._config["channels"] % 4 == 0, "# of channels must be multiple of 4"


class FilterOpTestModelGenerator(ImageInputOpTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config.update(
            {key: cfg.pop(key) for key in ["K_h", "K_w", "padding", "strides"]}
        )
        super()._set_config(cfg)

    def build(self) -> None:
        try:
            super().build()
        except ValueError as e:
            if e.args[0].startswith("Negative dimension size caused by"):
                raise ValueError(
                    "Negative dimension size (Hint: if using 'valid' padding "
                    "verify that the kernel is at least the size of input image)"
                ) from e
            else:
                raise


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


@pytest.mark.skip_on_device  # type: ignore
def test_converted_single_op_model(
    xcore_model: XCOREModel, converted_op_code: XCOREOpCodes
) -> None:
    operators = xcore_model.subgraphs[0].operators
    assert len(operators) == 1
    op = operators[0]
    assert op.operator_code.code is converted_op_code


@pytest.mark.skip_on_device  # type: ignore
def test_reference_model_regression(
    reference_model: XCOREModel, reference_op_code: ValidOpCodes
) -> None:
    operators = reference_model.subgraphs[0].operators
    assert len(operators) == 1
    op = operators[0]
    assert op.operator_code.code is reference_op_code
