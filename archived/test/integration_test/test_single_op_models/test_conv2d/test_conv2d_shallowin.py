# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from tflite2xcore.model_generation import Configuration

from . import Conv2dProperTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Conv2dShallowinTestModelGenerator(Conv2dProperTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        cfg.setdefault("input_channels", 4)
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        assert (
            self._config["K_w"] * self._config["input_channels"] <= 32
        ), "K_w * input_channels > 32 is reserved for general conv2d testing"
        assert (
            self._config["K_h"] != 1 or self._config["K_w"] != 1
        ), "1x1 kernel is reserved for conv2d_1x1 testing"


GENERATOR = Conv2dShallowinTestModelGenerator


if __name__ == "__main__":
    pytest.main()
