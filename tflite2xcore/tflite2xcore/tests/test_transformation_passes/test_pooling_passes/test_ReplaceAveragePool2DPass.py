# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ReplaceAveragePool2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_avgpool
from .conftest import (
    PARAMS,
    test_matching_params,
    test_non_matching_input_channels,
    test_non_matching_fused_activation,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model():
    return build_avgpool


@pytest.fixture()
def trf_pass():
    return ReplaceAveragePool2DPass()


if __name__ == "__main__":
    pytest.main()
