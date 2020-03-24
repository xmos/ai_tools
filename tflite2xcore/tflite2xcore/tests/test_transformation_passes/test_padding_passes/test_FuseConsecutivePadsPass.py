# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import FuseConsecutivePadsPass
from tflite2xcore.operator_codes import BuiltinOpCodes

from ..model_builders import build_pad, build_consecutive_pads
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    update_params_with_paddings
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = update_params_with_paddings(
    deepcopy(PARAMS),
    is_matching=lambda padding: True
)

# NOTE: this is intentional to keep test case count lower
PARAMS["default"]["paddings"] = PARAMS["smoke"]["paddings"]


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def build_model():
    return build_pad


@pytest.fixture()
def trf_pass():
    return FuseConsecutivePadsPass()


@pytest.fixture()
def model(input_shape, paddings, paddings_NC):
    return build_consecutive_pads(input_shape=input_shape,
                                  paddings_1=paddings,
                                  paddings_2=paddings_NC)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_non_matching_single_pad(trf_pass, input_shape, paddings):
    model = build_pad(input_shape=input_shape, paddings=paddings)
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
