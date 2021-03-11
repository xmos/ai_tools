# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    ParallelizeRequant16To8Pass,
)

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_requantize_16_to_8,
)

from .test_ParallelizeFullyConnectedPass import PARAMS
from .conftest import test_matching_params, test_mutate


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass(num_threads: int) -> XCOREModel:
    return ParallelizeRequant16To8Pass(num_threads=num_threads)


@pytest.fixture()
def model(outputs: int, input_channels: int) -> ModelTransformationPass:
    return build_XC_requantize_16_to_8(outputs=outputs, input_channels=input_channels)


if __name__ == "__main__":
    pytest.main()
