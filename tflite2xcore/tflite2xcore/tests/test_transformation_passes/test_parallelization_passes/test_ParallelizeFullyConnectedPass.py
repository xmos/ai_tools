# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    ParallelizeFullyConnectedPass,
)

from tflite2xcore.tests.test_transformation_passes.model_builders import build_XC_fc

from ..test_fully_connected_passes.conftest import PARAMS
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
    return ParallelizeFullyConnectedPass(num_threads=num_threads)


@pytest.fixture()
def model(outputs: int, input_channels: int) -> ModelTransformationPass:
    return build_XC_fc(outputs=outputs, input_channels=input_channels)


if __name__ == "__main__":
    pytest.main()
