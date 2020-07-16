# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ScratchMemoryFullyConnectedPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_fc_deepin_anyout,
)

from ..test_fully_connected_passes.conftest import PARAMS
from .conftest import test_matching_params, test_mutate


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return ScratchMemoryFullyConnectedPass()


@pytest.fixture()
def model(outputs, input_channels):
    return build_XC_fc_deepin_anyout(outputs=outputs, input_channels=input_channels)


if __name__ == "__main__":
    pytest.main()
