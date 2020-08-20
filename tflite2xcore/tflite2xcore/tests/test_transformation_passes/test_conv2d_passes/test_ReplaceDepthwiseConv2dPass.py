# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.xcore_schema import XCOREOpCodes
from tflite2xcore.transformation_passes import ReplaceDepthwiseConv2dPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_depthwise_conv2d,
)
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_tensors,
)
from .test_ReplaceDeepConv2dPass import test_mutate


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {"depth_multiplier": [1], "non_matching_depth_multiplier": [2, 5, 16]}
)

PARAMS["default"].update(
    {"depth_multiplier": [1], "non_matching_depth_multiplier": [2, 16]}
)

PARAMS["smoke"].update({"depth_multiplier": [1], "non_matching_depth_multiplier": [16]})


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
    return build_depthwise_conv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
    )


@pytest.fixture()
def custom_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_conv2d_depthwise


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_non_matching_input_channels(
    trf_pass,
    build_model,
    depth_multiplier,
    kernel_height,
    kernel_width,
    non_matching_input_channels,
    input_size,
    padding,
    strides,
):
    model = build_model(
        weight_shape=[
            kernel_height,
            kernel_width,
            non_matching_input_channels,
            depth_multiplier,
        ],
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_depth_multiplier(
    trf_pass,
    build_model,
    non_matching_depth_multiplier,
    kernel_height,
    kernel_width,
    input_channels,
    input_size,
    padding,
    strides,
):
    model = build_model(
        weight_shape=[
            kernel_height,
            kernel_width,
            input_channels,
            non_matching_depth_multiplier,
        ],
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
