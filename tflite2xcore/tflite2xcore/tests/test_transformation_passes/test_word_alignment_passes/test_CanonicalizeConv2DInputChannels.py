# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import CanonicalizeConv2DInputChannels

from tflite2xcore.tests.test_transformation_passes.model_builders import build_conv2d
from tflite2xcore.tests.test_transformation_passes.test_conv2d_passes.conftest import (
    PARAMS as CONV_PARAMS,
    test_non_matching_input_channels,
)
from ..conftest import (
    _test_non_matching_params,
    test_matching_params,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(CONV_PARAMS)

for k in PARAMS:
    PARAMS[k]["input_channels"] = deepcopy(
        CONV_PARAMS[k]["non_matching_input_channels"]
    )
    PARAMS[k]["non_matching_input_channels"] = deepcopy(
        CONV_PARAMS[k]["input_channels"]
    )
    PARAMS[k]["output_channels"] = deepcopy(
        CONV_PARAMS["smoke"]["output_channels"]
        + CONV_PARAMS["smoke"]["non_matching_output_channels"]
    )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return CanonicalizeConv2DInputChannels()


@pytest.fixture()
def build_model():
    return build_conv2d


@pytest.fixture()
def weight_shape(output_channels, kernel_height, kernel_width, input_channels):
    return [output_channels, kernel_height, kernel_width, input_channels]


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    model = build_conv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    return model


if __name__ == "__main__":
    pytest.main()
