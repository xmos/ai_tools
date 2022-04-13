# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from tflite2xcore.model_generation import Configuration

from . import Conv2dGenericTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class SinglePixelConv2dTestModelGenerator(Conv2dGenericTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        assert "height" not in cfg and "width" not in cfg, (
            "height and width should not be specified "
            "(they are inferred from kernel height and width)"
        )
        cfg["height"] = cfg.setdefault("K_h", 1)
        cfg["width"] = cfg.setdefault("K_w", 1)

        cfg.setdefault("padding", "valid")
        super()._set_config(cfg)

    def check_config(self) -> None:
        super().check_config()
        assert (
            self._config["padding"] == "valid"
        ), "Only valid padding is allowed in single pixel conv2d tests"


GENERATOR = SinglePixelConv2dTestModelGenerator



if __name__ == "__main__":
    pytest.main()
