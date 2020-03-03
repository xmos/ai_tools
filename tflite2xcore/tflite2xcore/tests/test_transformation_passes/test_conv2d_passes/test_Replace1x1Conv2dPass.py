# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import Replace1x1Conv2dPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_conv2d
from .conftest import (
    PARAMS,
    test_matching_params,
    test_non_matching_output_channels,
    test_non_matching_kernel_height,
    test_non_matching_kernel_width,
    test_non_matching_input_channels,
    test_non_matching_types
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update({
    "kernel_width": [1],
    "non_matching_kernel_width": [2, 3, 5],
    "kernel_height": [1],
    "non_matching_kernel_height": [2, 3, 5],
    "input_channels": [4, 8, 16, 32],
    "non_matching_input_channels": [3, 9, 15],
    "output_channels": [4, 8, 16, 32],
    "non_matching_output_channels": [3, 9, 15]
})

PARAMS["smoke"].update({
    "kernel_width": [1],
    "non_matching_kernel_width": [3, 5],
    "kernel_height": [1],
    "non_matching_kernel_height": [3, 5],
    "input_channels": [4, 32],
    "non_matching_input_channels": [3, 9],
    "output_channels": [4, 32],
    "non_matching_output_channels": [3, 9]
})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def build_model():
    return build_conv2d


@pytest.fixture()
def trf_pass():
    return Replace1x1Conv2dPass()


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    model = build_conv2d(weight_shape=weight_shape, input_size=input_size,
                         padding=padding, strides=strides)
    return model


if __name__ == "__main__":
    pytest.main()
