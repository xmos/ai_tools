# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from typing import Tuple

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    ScratchMemoryConv2d1x1Pass,
)

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_conv2d_1x1,
)

from ..test_conv2d_passes.test_Replace1x1Conv2dPass import PARAMS
from ..test_conv2d_passes.conftest import weight_shape
from .conftest import test_matching_params, test_mutate


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ModelTransformationPass:
    return ScratchMemoryConv2d1x1Pass()


@pytest.fixture()
def model(
    weight_shape: Tuple[int, int, int, int],
    input_size: Tuple[int, int],
    strides: Tuple[int, int],
) -> XCOREModel:
    return build_XC_conv2d_1x1(
        weight_shape=weight_shape, input_size=input_size, strides=strides
    )


if __name__ == "__main__":
    pytest.main()
