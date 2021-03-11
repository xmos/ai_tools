# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
from typing import Tuple
from copy import deepcopy

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import XCOREOpCodes, Padding
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    ReplaceDepthwiseConv2dPass,
)

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    ModelBuilder,
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
def build_model() -> ModelBuilder:
    return build_depthwise_conv2d


@pytest.fixture()
def trf_pass() -> ReplaceDepthwiseConv2dPass:
    return ReplaceDepthwiseConv2dPass()


@pytest.fixture()
def weight_shape(
    depth_multiplier: int, kernel_height: int, kernel_width: int, input_channels: int
) -> Tuple[int, int, int, int]:
    return [kernel_height, kernel_width, input_channels, depth_multiplier]


@pytest.fixture()
def model(
    weight_shape: Tuple[int, int, int, int],
    input_size: Tuple[int, int],
    padding: Padding,
    strides: Tuple[int, int],
) -> XCOREModel:
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
    trf_pass: ModelTransformationPass,
    build_model: ModelBuilder,
    depth_multiplier: int,
    kernel_height: int,
    kernel_width: int,
    non_matching_input_channels: int,
    input_size: Tuple[int, int],
    padding: Padding,
    strides: Tuple[int, int],
) -> None:
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
    trf_pass: ModelTransformationPass,
    build_model: ModelBuilder,
    non_matching_depth_multiplier: int,
    kernel_height: int,
    kernel_width: int,
    input_channels: int,
    input_size: Tuple[int, int],
    padding: Padding,
    strides: Tuple[int, int],
) -> None:
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
