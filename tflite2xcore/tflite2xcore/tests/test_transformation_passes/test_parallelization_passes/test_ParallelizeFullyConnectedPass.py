# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import ParallelizeFullyConnectedPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_fc_deepin_anyout,
)

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
def trf_pass(num_threads):
    return ParallelizeFullyConnectedPass(num_threads=num_threads)


@pytest.fixture()
def model(outputs, input_channels):
    return build_XC_fc_deepin_anyout(outputs=outputs, input_channels=input_channels)


if __name__ == "__main__":
    pytest.main()
