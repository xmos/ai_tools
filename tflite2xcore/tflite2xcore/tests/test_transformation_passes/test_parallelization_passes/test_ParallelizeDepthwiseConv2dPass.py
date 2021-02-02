# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from copy import deepcopy
from typing import Tuple

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    ParallelizeDepthwiseConv2dPass,
)

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_conv2d_depthwise,
)

from ..test_conv2d_passes.test_ReplaceDepthwiseConv2dPass import PARAMS, weight_shape
from .conftest import test_matching_params, test_mutate, PARAMS as PAR_PARAMS


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

for k in PARAMS:
    PARAMS[k].update({"num_threads": PAR_PARAMS[k]["num_threads"]})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass(num_threads: int) -> ModelTransformationPass:
    return ParallelizeDepthwiseConv2dPass(num_threads=num_threads)


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
