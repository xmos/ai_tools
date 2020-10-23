# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import tensorflow as tf

from tflite2xcore.xcore_schema import ExternalOpCodes, XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore.model_generation import Configuration
from tflite2xcore.model_generation.data_factories import InputInitializerDataFactory

from . import (
    BinarizedTestRunner,
    LarqCompositeTestModelGenerator,
    LarqConverter,
)

from . import (
    test_reference_model_regression,
    test_converted_single_op_model,
    test_mean_abs_diffs,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class BSignTestModelGenerator(LarqCompositeTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        for key in ("K_w", "K_h", "output_channels"):
            assert key not in cfg, f"{key} should not be specified for bsing tests"
        cfg["output_channels"] = 32
        cfg["K_w"] = cfg["K_h"] = 1
        super()._set_config(cfg)


GENERATOR = BSignTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


class BSignTestRunner(BinarizedTestRunner):
    def make_lce_converter(self) -> LarqConverter:
        return LarqConverter(self, self.get_built_model, remove_last_op=True)

    def make_repr_data_factory(self) -> InputInitializerDataFactory:
        return InputInitializerDataFactory(
            self, lambda: self._model_generator.input_shape, dtype=tf.int8
        )


RUNNER = BSignTestRunner

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def reference_op_code() -> ExternalOpCodes:
    return ExternalOpCodes.LceQuantize


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_bsign_8


@pytest.fixture  # type: ignore
def mean_abs_diff_tolerance() -> float:
    return 0.0


if __name__ == "__main__":
    pytest.main()
