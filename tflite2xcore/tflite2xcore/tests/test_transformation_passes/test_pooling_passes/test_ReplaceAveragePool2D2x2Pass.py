# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.xcore_schema import Padding
from tflite2xcore.transformation_passes import ReplaceAveragePool2D2x2Pass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_avgpool
from .conftest import (
    PARAMS,
    test_matching_params,
    test_non_matching_input_channels,
    test_non_matching_fused_activation,
    test_non_matching_input_height,
    test_non_matching_input_width,
    test_non_matching_pool_h,
    test_non_matching_pool_w,
    test_non_matching_stride_h,
    test_non_matching_stride_w,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update(
    {
        "input_height": [2, 4, 8, 12],
        "non_matching_input_height": [3, 9, 13, 23],
        "input_width": [2, 4, 8, 12],
        "non_matching_input_width": [3, 9, 13, 23],
        "padding": Padding,
        "stride_h": [2],
        "non_matching_stride_h": [1, 3],
        "stride_w": [2],
        "non_matching_stride_w": [1, 3],
        "pool_h": [2],
        "non_matching_pool_h": [1, 3],
        "pool_w": [2],
        "non_matching_pool_w": [1, 3],
    }
)

PARAMS["smoke"].update(
    {
        "input_height": [2, 12],
        "non_matching_input_height": [3, 13],
        "input_width": [2, 12],
        "non_matching_input_width": [3, 13],
        "padding": Padding,
        "stride_h": [2],
        "non_matching_stride_h": [3],
        "stride_w": [2],
        "non_matching_stride_w": [3],
        "pool_h": [2],
        "non_matching_pool_h": [3],
        "pool_w": [2],
        "non_matching_pool_w": [3],
    }
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model():
    return build_avgpool


@pytest.fixture()
def trf_pass():
    return ReplaceAveragePool2D2x2Pass()


if __name__ == "__main__":
    pytest.main()
