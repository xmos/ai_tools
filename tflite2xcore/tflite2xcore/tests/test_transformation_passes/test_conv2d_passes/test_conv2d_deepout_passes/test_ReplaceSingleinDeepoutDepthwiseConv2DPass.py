# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import ReplaceSingleinDeepoutDepthwiseConv2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_depthwise_conv2d
from . import test_ReplaceShallowinDeepoutConv2DPass
from .conftest import (
    PARAMS,
    _test_non_matching_dim,
    test_matching_params,
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
def build_model():
    return build_depthwise_conv2d


@pytest.fixture()
def trf_pass():
    return ReplaceSingleinDeepoutDepthwiseConv2DPass()


@pytest.fixture()
def weight_shape(depth_multiplier, kernel_height, kernel_width, input_channels):
    return [kernel_height, kernel_width, input_channels, depth_multiplier]


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    print(weight_shape)
    return build_depthwise_conv2d(weight_shape=weight_shape, input_size=input_size,
                                  padding=padding, strides=strides)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

# TODO refactor these to share with other depthwise conv tests
def test_non_matching_depth_multiplier(trf_pass, build_model,
                                       weight_shape, input_size, padding, strides,
                                       non_matching_depth_multiplier):
    weight_shape[3] = non_matching_depth_multiplier
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


def test_non_matching_kernel_height(trf_pass, build_model,
                                    weight_shape, input_size, padding, strides,
                                    non_matching_kernel_height):
    weight_shape[0] = non_matching_kernel_height
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


def test_non_matching_kernel_width(trf_pass, build_model,
                                   weight_shape, input_size, padding, strides,
                                   non_matching_kernel_width):
    weight_shape[1] = non_matching_kernel_width
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


def test_non_matching_input_channels(trf_pass, build_model,
                                     weight_shape, input_size, padding, strides,
                                     non_matching_input_channels):
    weight_shape[2] = non_matching_input_channels
    _test_non_matching_dim(trf_pass, build_model, weight_shape, input_size, padding, strides)


if __name__ == "__main__":
    pytest.main()
