# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from copy import deepcopy
from typing import Tuple

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    ParallelizeGlobalAveragePool2DPass,
)

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_avgpool2d_global,
)

from ..test_pooling_passes.test_ReplaceGlobalAveragePool2DPass import PARAMS
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
def trf_pass(num_threads: int) -> XCOREModel:
    return ParallelizeGlobalAveragePool2DPass(num_threads=num_threads)


@pytest.fixture()
def model(
    input_shape: Tuple[int, int, int], reduction_dims: Tuple[int, ...]
) -> ModelTransformationPass:
    return build_XC_avgpool2d_global(
        input_shape=input_shape, reduction_dims=reduction_dims
    )


if __name__ == "__main__":
    pytest.main()
