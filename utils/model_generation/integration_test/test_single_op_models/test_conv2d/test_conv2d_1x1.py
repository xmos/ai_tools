# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore

from tflite2xcore._model_generation import Configuration

from . import Conv2dGenericTestModelGenerator, test_output


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Conv2d1x1TestModelGenerator(Conv2dGenericTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        assert cfg.setdefault("K_h", 1) == 1, "Kernel height must be 1"
        assert cfg.setdefault("K_w", 1) == 1, "Kernel width must be 1"
        super()._set_config(cfg)


GENERATOR = Conv2d1x1TestModelGenerator

#  ----------------------------------------------------------------------------
#                                   CONFIGS
#  ----------------------------------------------------------------------------

CONFIGS = {
    "default": [
        {},
        {"height": 2, "width": 2},
        {"height": 3, "width": 6},
        {"input_channels": 8, "output_channels": 12},
        {"input_channels": 32, "output_channels": 16},
        {"height": 8, "width": 2, "input_channels": 12, "output_channels": 64},
        {
            "weight_init": ("Constant", 1.0),
            "bias_init": ("Constant", 1.0),
            "input_init": ("Constant", 1.0),
        },
        {
            "height": 1,
            "width": 4,
            "weight_init": ("RandomNormal", 0.0, 0.1),
            "bias_init": ("RandomNormal", -0.5, 0.02),
            "input_init": ("RandomUniform", 10.0),
        },
        {"num_threads": 2},
        {"num_threads": 5},
    ],
}


if __name__ == "__main__":
    pytest.main()
