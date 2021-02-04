# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from copy import deepcopy
from typing import Tuple, Callable

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    ParallelizeConv2dPass,
)

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_conv2d_deep,
    build_XC_conv2d_shallowin,
    build_XC_conv2d_1x1,
)

from ..test_conv2d_passes.conftest import PARAMS, weight_shape
from .conftest import test_matching_params, test_mutate, PARAMS as PAR_PARAMS


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

for k in PARAMS:
    PARAMS[k].update(
        {
            "num_threads": PAR_PARAMS[k]["num_threads"],
            "model_builder": [
                build_XC_conv2d_deep,
                build_XC_conv2d_shallowin,
                build_XC_conv2d_1x1,
            ],
        }
    )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass(num_threads: int) -> ModelTransformationPass:
    return ParallelizeConv2dPass(num_threads=num_threads)


@pytest.fixture()
def model(
    model_builder: Callable[..., XCOREModel],
    weight_shape: Tuple[int, int, int, int],
    input_size: Tuple[int, int],
    strides: Tuple[int, int],
) -> XCOREModel:
    return model_builder(
        weight_shape=weight_shape, input_size=input_size, strides=strides
    )


if __name__ == "__main__":
    pytest.main()
