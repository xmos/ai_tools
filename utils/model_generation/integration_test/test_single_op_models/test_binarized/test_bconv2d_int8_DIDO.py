# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_schema import ExternalOpCodes, XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore.model_generation import Configuration

from . import BConv2dGenericTestModelGenerator

from .test_bconv2d_int8 import RUNNER  # pylint: disable=unused-import
from . import (  # pylint: disable=unused-import
    test_reference_model_regression,
    test_converted_single_op_model,
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class BConv2dInt8DeepInDeepOutTestModelGenerator(BConv2dGenericTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        cfg.setdefault("padding", "valid")
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        assert (
            self._config["input_channels"] % 256 == 0
        ), "# of input channels must be multiple of 256"
        assert (
            self._config["output_channels"] % 16 == 0
        ), "# of input channels must be multiple of 16"


GENERATOR = BConv2dInt8DeepInDeepOutTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def bitpacked_outputs() -> bool:
    return False


@pytest.fixture  # type: ignore
def reference_op_code() -> ExternalOpCodes:
    return ExternalOpCodes.LceBconv2d


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_bconv2d_int8_DIDO


if __name__ == "__main__":
    pytest.main()
