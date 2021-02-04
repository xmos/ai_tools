# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest
import tensorflow as tf
import numpy as np

from tflite2xcore.xcore_schema import ExternalOpCodes, XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore.model_generation import Configuration
from tflite2xcore.model_generation.data_factories import InputInitializerDataFactory

from . import (
    BinarizedSingleOpRunner,
    LarqCompositeTestModelGenerator,
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


class BSignTestModelGenerator(LarqCompositeTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        for key in ("K_w", "K_h", "output_channels", "activation"):
            assert key not in cfg, f"{key} should not be specified for bsign tests"
        cfg["output_channels"] = 32
        cfg["K_w"] = cfg["K_h"] = 1
        super()._set_config(cfg)


GENERATOR = BSignTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


class BSignTestRunner(BinarizedSingleOpRunner):
    def make_lce_converter(self) -> LarqSingleOpConverter:
        return LarqSingleOpConverter(self, self.get_built_model, remove_last_op=True)

    def _set_config(self, cfg: Configuration) -> None:
        cfg["input_range"] = cfg.pop(
            "input_range", (np.iinfo(np.int8).min, np.iinfo(np.int8).max)
        )
        assert (
            "output_range" not in cfg
        ), f"output_range cannot be specified for Bsign tests"
        super()._set_config(cfg)

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


if __name__ == "__main__":
    pytest.main()
