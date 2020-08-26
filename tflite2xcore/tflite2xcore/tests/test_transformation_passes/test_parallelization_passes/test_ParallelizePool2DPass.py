# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy
from typing import Tuple, Callable

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    ParallelizePooling2DPass,
)

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_maxpool2d,
    build_XC_avgpool2d,
)

from ..test_pooling_passes.conftest import PARAMS, pool_size
from .conftest import test_matching_params, test_mutate, PARAMS as PAR_PARAMS


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

for k in PARAMS:
    PARAMS[k].update(
        {
            "num_threads": PAR_PARAMS[k]["num_threads"],
            "model_builder": [build_XC_maxpool2d, build_XC_avgpool2d],
        }
    )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass(num_threads: int) -> ModelTransformationPass:
    return ParallelizePooling2DPass(num_threads=num_threads)


@pytest.fixture()
def model(
    model_builder: Callable[..., XCOREModel],
    input_shape: Tuple[int, int, int],
    pool_size: Tuple[int, int],
    strides: Tuple[int, int],
) -> XCOREModel:
    return model_builder(input_shape=input_shape, pool_size=pool_size, strides=strides)


if __name__ == "__main__":
    pytest.main()
