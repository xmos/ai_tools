# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
from typing import Tuple
from copy import deepcopy

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import XCOREOpCodes, Padding
from tflite2xcore.transformation_passes import Replace1x1Conv2dPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_conv2d,
    ModelBuilder,
)
from .conftest import (
    PARAMS,
    test_replace_mutate as test_mutate,
    test_matching_params,
    test_non_matching_output_channels,
    test_non_matching_kernel_height,
    test_non_matching_kernel_width,
    test_non_matching_input_channels,
    test_non_matching_stride_h,
    test_non_matching_stride_w,
    test_non_matching_tensors,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {
        "kernel_width": [1],
        "non_matching_kernel_width": [2, 3, 5, 7],
        "kernel_height": [1],
        "non_matching_kernel_height": [2, 3, 5, 7],
        "stride_h": [1],
        "non_matching_stride_h": [2, 3],
        "stride_w": [1],
        "non_matching_stride_w": [2, 3],
    }
)

PARAMS["default"].update(
    {
        "kernel_width": [1],
        "non_matching_kernel_width": [2, 3, 7],
        "kernel_height": [1],
        "non_matching_kernel_height": [2, 3, 7],
        "stride_h": [1],
        "non_matching_stride_h": [2, 3],
        "stride_w": [1],
        "non_matching_stride_w": [2, 3],
    }
)

PARAMS["smoke"].update(
    {
        "kernel_width": [1],
        "non_matching_kernel_width": [2, 3],
        "kernel_height": [1],
        "non_matching_kernel_height": [2, 3],
        "stride_h": [1],
        "non_matching_stride_h": [2],
        "stride_w": [1],
        "non_matching_stride_w": [2],
    }
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model() -> ModelBuilder:
    return build_conv2d


@pytest.fixture()
def trf_pass() -> Replace1x1Conv2dPass:
    return Replace1x1Conv2dPass()


@pytest.fixture()
def model(
    weight_shape: Tuple[int, int, int, int],
    input_size: Tuple[int, int],
    padding: Padding,
    strides: Tuple[int, int],
) -> XCOREModel:
    model = build_conv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    return model


@pytest.fixture()
def custom_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_conv2d_1x1


if __name__ == "__main__":
    pytest.main()
