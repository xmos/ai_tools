# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
import itertools

from copy import deepcopy

from tflite2xcore.transformation_passes import ReplaceGlobalAveragePool2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_mean
from .conftest import PARAMS, test_matching_params, _test_non_matching_params


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update({"reduction_dims": [(1, 2), (2, 1)]})
PARAMS["default"].update(
    {
        "non_matching_reduction_dims": [
            t
            for t in itertools.chain(
                itertools.product(range(4)),
                itertools.product(range(4), repeat=2),
                itertools.product(range(4), repeat=3),
            )
            if t not in PARAMS["default"]["reduction_dims"]
        ]
    }
)

PARAMS["smoke"].update({"reduction_dims": PARAMS["default"]["reduction_dims"]})
PARAMS["smoke"].update(
    {
        "non_matching_reduction_dims": [
            t
            for t in itertools.chain(
                itertools.product(range(1, 4)), itertools.product(range(1, 4), repeat=2)
            )
            if t not in PARAMS["smoke"]["reduction_dims"]
        ]
    }
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return ReplaceGlobalAveragePool2DPass()


@pytest.fixture()
def model(input_shape, reduction_dims):
    return build_mean(input_shape=input_shape, reduction_dims=reduction_dims)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_non_matching_input_channels(
    trf_pass, input_size, non_matching_input_channels, reduction_dims
):
    input_shape = (*input_size, non_matching_input_channels)
    model = build_mean(input_shape=input_shape, reduction_dims=reduction_dims)
    _test_non_matching_params(trf_pass, model)


def test_non_matching_reduction_dims(
    trf_pass, input_shape, non_matching_reduction_dims
):
    model = build_mean(
        input_shape=input_shape, reduction_dims=non_matching_reduction_dims
    )
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
