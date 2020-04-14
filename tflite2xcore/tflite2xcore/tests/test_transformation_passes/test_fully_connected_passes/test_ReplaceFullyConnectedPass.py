# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ReplaceFullyConnectedPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_fc
)
from .conftest import (
    PARAMS,
    test_matching_params,
    test_non_matching_tensors
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def trf_pass():
    return ReplaceFullyConnectedPass()


@pytest.fixture()
def model(input_shape, outputs):
    return build_fc(input_shape=input_shape, outputs=outputs)


if __name__ == "__main__":
    pytest.main()
