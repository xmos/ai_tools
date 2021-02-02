# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from tflite2xcore.transformation_passes import ReplaceMaxPool2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_maxpool,
    ModelBuilder,
)
from .test_ReplaceMaxPool2D2x2Pass import custom_opcode
from .conftest import (
    PARAMS,
    test_matching_params,
    test_mutate,
    test_non_matching_input_channels,
    test_non_matching_fused_activation,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model() -> ModelBuilder:
    return build_maxpool


@pytest.fixture()
def trf_pass() -> ReplaceMaxPool2DPass:
    return ReplaceMaxPool2DPass()


if __name__ == "__main__":
    pytest.main()
