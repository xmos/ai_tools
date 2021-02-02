# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from tflite2xcore.transformation_passes import ReplaceAveragePool2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_avgpool,
    ModelBuilder,
)
from .test_ReplaceAveragePool2D2x2Pass import custom_opcode
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
    return build_avgpool


@pytest.fixture()
def trf_pass() -> ReplaceAveragePool2DPass:
    return ReplaceAveragePool2DPass()


if __name__ == "__main__":
    pytest.main()
