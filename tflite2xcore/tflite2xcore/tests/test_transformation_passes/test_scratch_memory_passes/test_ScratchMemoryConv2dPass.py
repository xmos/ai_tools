# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy
from typing import Tuple, Callable

from tflite2xcore.pass_manager import ModelTransformationPass
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import ScratchMemoryConv2dPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_conv2d_deep,
    build_XC_conv2d_shallowin,
)

from ..test_conv2d_passes.conftest import PARAMS, weight_shape
from .conftest import test_matching_params, test_mutate


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

for k in PARAMS:
    PARAMS[k].update(
        {"model_builder": [build_XC_conv2d_deep, build_XC_conv2d_shallowin]}
    )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ModelTransformationPass:
    return ScratchMemoryConv2dPass()


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
