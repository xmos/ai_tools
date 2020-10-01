# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore

from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore._model_generation import Configuration

from . import Conv2dProperTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Conv2dTestModelGenerator(Conv2dProperTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        cfg.setdefault("input_channels", 20)
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        assert (
            self._config["K_w"] * self._config["input_channels"] > 32
        ), "K_w * input_channels <= 32 is reserved for conv2d_shallowin testing"
        assert (
            self._config["K_h"] != 1 or self._config["K_w"] != 1
        ), "1x1 kernel is reserved for conv2d_1x1 testing"


GENERATOR = Conv2dTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_conv2d_deep


if __name__ == "__main__":
    pytest.main()
