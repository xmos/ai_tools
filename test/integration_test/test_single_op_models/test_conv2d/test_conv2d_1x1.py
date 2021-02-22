# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import pytest

from tflite2xcore.xcore_schema import XCOREOpCodes
from tflite2xcore.model_generation import Configuration

from . import Conv2dProperTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_reference_model_regression,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Conv2d1x1TestModelGenerator(Conv2dProperTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        cfg.setdefault("K_h", 1)
        cfg.setdefault("K_w", 1)
        cfg.setdefault("strides", (1, 1))
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        assert self._config["K_h"] == 1, "Kernel height must be 1"
        assert self._config["K_w"] == 1, "Kernel width must be 1"
        assert self._config["strides"] == (1, 1), "strides must be (1, 1)"


GENERATOR = Conv2d1x1TestModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_conv2d_1x1


if __name__ == "__main__":
    pytest.main()
