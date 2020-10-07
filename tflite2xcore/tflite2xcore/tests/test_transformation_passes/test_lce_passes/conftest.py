# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
from copy import deepcopy

from tflite2xcore.xcore_schema import Padding

from ..model_builders import build_lceBconv2d
from ..conftest import (
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_tensors,
)
from ..test_conv2d_passes.conftest import (
    PARAMS,
    weight_shape,
    test_non_matching_input_channels,
    test_non_matching_output_channels,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {
        "input_channels": [256, 512, 1024],
        "non_matching_input_channels": [32, 47, 128],
        "output_channels": [32, 64, 256],
        "non_matching_output_channels": [16, 31, 48],
    }
)

for key in (
    "input_channels",
    "non_matching_input_channels",
    "output_channels",
    "non_matching_output_channels",
):
    PARAMS["default"][key] = PARAMS["extended"][key][:-1]
    PARAMS["smoke"][key] = PARAMS["default"][key][:-1]


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model():
    return build_lceBconv2d

