# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import ReplaceDepthwiseConv2dPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_depthwise_conv2d
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_types
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update({
    "stride_h": [1, 2, 3],  # TODO: this should be the default after the conv2d improvements
    "stride_w": [1, 2, 3],  # TODO: this should be the default after the conv2d improvements
    "depth_multiplier": [1],
    "non_matching_depth_multiplier": [2, 3, 4, 16]
})

PARAMS["smoke"].update({
    "stride_h": [1, 2],  # TODO: this should be the default after the conv2d improvements
    "stride_w": [1, 2],  # TODO: this should be the default after the conv2d improvements
    "depth_multiplier": [1],
    "non_matching_depth_multiplier": [2, 16]
})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model():
    return build_depthwise_conv2d


@pytest.fixture()
def trf_pass():
    return ReplaceDepthwiseConv2dPass()


@pytest.fixture()
def weight_shape(depth_multiplier, kernel_height, kernel_width, input_channels):
    return [kernel_height, kernel_width, input_channels, depth_multiplier]


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    return build_depthwise_conv2d(weight_shape=weight_shape, input_size=input_size,
                                  padding=padding, strides=strides)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_non_matching_depth_multiplier(trf_pass, build_model,
                                       weight_shape, input_size, padding, strides,
                                       non_matching_depth_multiplier):
    weight_shape[3] = non_matching_depth_multiplier
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        padding=padding, strides=strides)
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
