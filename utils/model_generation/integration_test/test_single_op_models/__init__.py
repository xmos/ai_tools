# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
from abc import abstractmethod

from tflite2xcore._model_generation import Configuration
from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this

from .. import IntegrationTestModelGenerator, test_output


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class FilterOpTestModelGenerator(IntegrationTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config.update(
            dict(
                K_h=cfg.pop("K_h"),
                K_w=cfg.pop("K_w"),
                height=cfg.pop("height"),
                width=cfg.pop("width"),
                padding=cfg.pop("padding", "same"),
                strides=cfg.pop("strides", (1, 1)),
            )
        )
        super()._set_config(cfg)

    @abstractmethod
    def _build_core_model(self) -> tf.keras.Model:
        raise NotImplementedError()

    def build(self) -> None:
        self._prep_backend()
        try:
            self._model = self._build_core_model()
        except ValueError as e:
            if e.args[0].startswith("Negative dimension size caused by"):
                raise ValueError(
                    "Negative dimension size (Hint: if using 'valid' padding "
                    "verify that the kernel is at least the size of input image)"
                ) from e
            else:
                raise
        self._model.build()


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_converted_single_op_model(
    xcore_model: XCOREModel, converted_op_code: XCOREOpCodes
) -> None:
    operators = xcore_model.subgraphs[0].operators
    assert len(operators) == 1
    op = operators[0]
    assert op.operator_code.code is converted_op_code
