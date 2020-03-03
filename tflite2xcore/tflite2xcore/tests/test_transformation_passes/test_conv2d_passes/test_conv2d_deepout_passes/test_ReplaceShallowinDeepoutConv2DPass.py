# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import ReplaceShallowinDeepoutConv2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_conv2d
from .conftest import (
    PARAMS,
    test_matching_params,
    test_non_matching_output_channels,
    test_non_matching_kernel_height,
    test_non_matching_kernel_width,
    test_non_matching_input_channels,
    test_non_matching_types,
    test_non_matching_stride_w,
    test_non_matching_stride_h
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update({
    "non_matching_kernel_width": [2, 4, 6, 9],
    "input_channels": [1, 2, 3, 4],
    "non_matching_input_channels": [5, 16, 32]
})

PARAMS["smoke"].update({
    "non_matching_kernel_width": [2, 9],
    "input_channels": [1, 4],
    "non_matching_input_channels": [5, 32]
})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def build_model():
    return build_conv2d


@pytest.fixture()
def trf_pass():
    return ReplaceShallowinDeepoutConv2DPass()


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    return build_conv2d(weight_shape=weight_shape, input_size=input_size,
                        padding=padding, strides=strides)


if __name__ == "__main__":
    pytest.main()
