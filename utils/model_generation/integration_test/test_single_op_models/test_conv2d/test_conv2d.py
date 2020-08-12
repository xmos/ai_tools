# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore

from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore._model_generation import Configuration

from . import (
    Conv2dGenericTestModelGenerator,
    test_output,
    test_converted_single_op_model,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Conv2dTestModelGenerator(Conv2dGenericTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        input_channels = cfg.setdefault("input_channels", 20)
        try:
            assert (
                cfg["K_w"] * input_channels > 32
            ), "K_w * input_channels <= 32 is reserved for conv2d_shallowin testing"
            assert (
                cfg["K_h"] != 1 or cfg["K_w"] != 1
            ), "1x1 kernel is reserved for conv2d_1x1 testing"
        except KeyError:
            pass
        super()._set_config(cfg)


GENERATOR = Conv2dTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_conv2d_deep


if __name__ == "__main__":
    pytest.main()
