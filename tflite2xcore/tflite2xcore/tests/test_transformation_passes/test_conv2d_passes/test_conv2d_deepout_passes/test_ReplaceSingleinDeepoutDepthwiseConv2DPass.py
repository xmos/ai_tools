# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import ReplaceSingleinDeepoutDepthwiseConv2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_depthwise_conv2d
from . import test_ReplaceShallowinDeepoutConv2DPass
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
    "input_channels": [1],
    "non_matching_input_channels": [2, 3, 4, 5, 8, 16, 32],
    "non_matching_kernel_width":
        test_ReplaceShallowinDeepoutConv2DPass.PARAMS["default"]["non_matching_kernel_width"]
})

PARAMS["smoke"].update({
    "input_channels": [1],
    "non_matching_input_channels": [2, 16],
    "non_matching_kernel_width":
        test_ReplaceShallowinDeepoutConv2DPass.PARAMS["smoke"]["non_matching_kernel_width"]
})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def build_model():
    return build_depthwise_conv2d


@pytest.fixture()
def trf_pass():
    return ReplaceSingleinDeepoutDepthwiseConv2DPass()


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    return build_depthwise_conv2d(weight_shape=weight_shape, input_size=input_size,
                                  padding=padding, strides=strides)


if __name__ == "__main__":
    pytest.main()
