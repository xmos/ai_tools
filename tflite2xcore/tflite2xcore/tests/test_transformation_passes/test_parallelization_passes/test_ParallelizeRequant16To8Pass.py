# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest


from tflite2xcore.transformation_passes import ParallelizeRequant16To8Pass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_requantize_16_to_8,
)

from .test_ParallelizeFullyConnectedPass import PARAMS
from .conftest import test_matching_params, test_mutate


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass(num_threads):
    return ParallelizeRequant16To8Pass(num_threads=num_threads)


@pytest.fixture()
def model(outputs, input_channels):
    return build_XC_requantize_16_to_8(outputs=outputs, input_channels=input_channels)


if __name__ == "__main__":
    pytest.main()
