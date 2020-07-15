# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import ParallelizeConv2dPass

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
def trf_pass(num_threads):
    return ParallelizeConv2dPass(num_threads=num_threads)


@pytest.fixture()
def model(model_builder, weight_shape, input_size, strides):
    return model_builder(
        weight_shape=weight_shape, input_size=input_size, strides=strides
    )


if __name__ == "__main__":
    pytest.main()
