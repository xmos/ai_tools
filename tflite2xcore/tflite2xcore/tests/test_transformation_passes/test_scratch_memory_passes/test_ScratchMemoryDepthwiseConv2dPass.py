# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ScratchMemoryDepthwiseConv2dPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_conv2d_depthwise,
)

from ..test_conv2d_passes.test_ReplaceDepthwiseConv2dPass import PARAMS, weight_shape
from .conftest import test_matching_params, test_mutate


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return ScratchMemoryDepthwiseConv2dPass()


@pytest.fixture()
def model(weight_shape, input_size, strides):
    return build_XC_conv2d_depthwise(
        weight_shape=weight_shape, input_size=input_size, strides=strides
    )


if __name__ == "__main__":
    pytest.main()
