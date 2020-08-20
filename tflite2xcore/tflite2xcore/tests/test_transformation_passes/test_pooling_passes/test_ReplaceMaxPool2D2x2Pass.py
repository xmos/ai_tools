# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.xcore_schema import XCOREOpCodes
from tflite2xcore.transformation_passes import ReplaceMaxPool2D2x2Pass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_maxpool
from .test_ReplaceAveragePool2D2x2Pass import PARAMS
from .conftest import (
    test_matching_params,
    test_mutate,
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
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model():
    return build_maxpool


@pytest.fixture()
def trf_pass():
    return ReplaceMaxPool2D2x2Pass()


@pytest.fixture()
def custom_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_maxpool2d


if __name__ == "__main__":
    pytest.main()
