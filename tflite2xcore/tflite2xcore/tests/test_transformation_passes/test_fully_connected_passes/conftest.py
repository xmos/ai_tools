# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.tests.test_transformation_passes.model_builders import build_fc
from ..conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_tensors,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {"input_channels": [5, 8, 10, 16, 29, 64], "outputs": [1, 2, 10, 16, 29, 100]}
)

PARAMS["default"].update(
    {"input_channels": [5, 10, 29, 64], "outputs": [2, 10, 16, 100]}
)

PARAMS["smoke"].update({"input_channels": [5, 29], "outputs": [2, 10]})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def model(input_shape, outputs):
    return build_fc(input_shape=input_shape, outputs=outputs)
