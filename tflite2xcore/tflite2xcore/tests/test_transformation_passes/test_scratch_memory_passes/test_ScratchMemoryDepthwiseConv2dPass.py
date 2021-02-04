# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from typing import Tuple

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    ScratchMemoryDepthwiseConv2dPass,
)

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_conv2d_depthwise,
)

from ..test_conv2d_passes.test_ReplaceDepthwiseConv2dPass import PARAMS, weight_shape
from .conftest import test_matching_params, test_mutate


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ModelTransformationPass:
    return ScratchMemoryDepthwiseConv2dPass()


@pytest.fixture()
def model(
    weight_shape: Tuple[int, int, int, int],
    input_size: Tuple[int, int],
    strides: Tuple[int, int],
) -> XCOREModel:
    return build_XC_conv2d_depthwise(
        weight_shape=weight_shape, input_size=input_size, strides=strides
    )


if __name__ == "__main__":
    pytest.main()
