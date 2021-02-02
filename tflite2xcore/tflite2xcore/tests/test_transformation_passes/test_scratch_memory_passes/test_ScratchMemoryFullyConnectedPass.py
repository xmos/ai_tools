# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    ScratchMemoryFullyConnectedPass,
)

from tflite2xcore.tests.test_transformation_passes.model_builders import build_XC_fc

from ..test_fully_connected_passes.conftest import PARAMS
from .conftest import test_matching_params, test_mutate


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ModelTransformationPass:
    return ScratchMemoryFullyConnectedPass()


@pytest.fixture()
def model(outputs: int, input_channels: int) -> XCOREModel:
    return build_XC_fc(outputs=outputs, input_channels=input_channels)


if __name__ == "__main__":
    pytest.main()
