# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore

from ..test_conv2d import converted_op_code, Conv2dTestModelGenerator
from . import (
    FusedCustomReluMixin,
    # test_output,  # TODO: enable
    # test_converted_single_op_model,  # TODO: enable
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class CustomReluConv2dTestModelGenerator(
    FusedCustomReluMixin, Conv2dTestModelGenerator
):
    pass


GENERATOR = CustomReluConv2dTestModelGenerator

# TODO: add configs in yml
CONFIGS = {
    "default": {
        0: {
            "max_value": 4,
            "input_channels": 12,
            "output_channels": 32,
            "weight_init": ("RandomUniform", -1, 1),
            "bias_init": ("Constant", 0),
            "K_h": 1,
            "K_w": 4,
            "padding": "same",
            "strides": [1, 2],
            "height": 1,
            "width": 8,
            "input_init": ("RandomUniform", -1, 1),
            "num_threads": 1,
        }
    }
}


if __name__ == "__main__":
    pytest.main()
