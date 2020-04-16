# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import ReplaceSingleinDeepoutDepthwiseConv2DPass

from . import test_ReplaceShallowinDeepoutConv2DPass
from ..test_ReplaceDepthwiseConv2dPass import (
    build_depthwise_conv2d, build_model, weight_shape, model,
    test_non_matching_depth_multiplier
)
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_tensors,
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
    "depth_multiplier": PARAMS["default"]["output_channels"],
    "non_matching_depth_multiplier": PARAMS["default"]["non_matching_output_channels"],
    "non_matching_kernel_width":
        test_ReplaceShallowinDeepoutConv2DPass.PARAMS["default"]["non_matching_kernel_width"]
})

PARAMS["smoke"].update({
    "input_channels": [1],
    "non_matching_input_channels": [2, 16],
    "depth_multiplier": PARAMS["smoke"]["output_channels"],
    "non_matching_depth_multiplier": PARAMS["smoke"]["non_matching_output_channels"],
    "non_matching_kernel_width":
        test_ReplaceShallowinDeepoutConv2DPass.PARAMS["smoke"]["non_matching_kernel_width"]
})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def trf_pass():
    return ReplaceSingleinDeepoutDepthwiseConv2DPass()


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_non_matching_kernel_height(trf_pass, build_model,
                                    weight_shape, input_size, padding, strides,
                                    non_matching_kernel_height):
    weight_shape[0] = non_matching_kernel_height
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        padding=padding, strides=strides)
    _test_non_matching_params(trf_pass, model)


def test_non_matching_kernel_width(trf_pass, build_model,
                                   weight_shape, input_size, padding, strides,
                                   non_matching_kernel_width):
    weight_shape[1] = non_matching_kernel_width
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        padding=padding, strides=strides)
    _test_non_matching_params(trf_pass, model)


def test_non_matching_input_channels(trf_pass, build_model,
                                     weight_shape, input_size, padding, strides,
                                     non_matching_input_channels):
    weight_shape[2] = non_matching_input_channels
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        padding=padding, strides=strides)
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
