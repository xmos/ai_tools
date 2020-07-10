# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import itertools
from copy import deepcopy
from itertools import product
import numpy as np

from tflite2xcore.tests.test_transformation_passes.model_builders import build_fc
from ..conftest import (
    PARAMS,
    _test_non_matching_params,
    _make_name_type_pairs,
    NON_INT8_TEST_TYPES,
    NON_INT32_TEST_TYPES,
    test_matching_params,
    test_non_matching_tensors,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

_NON_MATCHING_TENSORS = list(
    itertools.chain(
        _make_name_type_pairs("input", NON_INT8_TEST_TYPES),
        _make_name_type_pairs("weights", NON_INT8_TEST_TYPES),
        _make_name_type_pairs("biases", NON_INT32_TEST_TYPES),
        _make_name_type_pairs("output", NON_INT8_TEST_TYPES),
    )
)

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {
        "input_channels": [5, 8, 10, 16, 29, 64],
        "outputs": [1, 2, 10, 16, 29, 100],
        "non_matching_tensors": _NON_MATCHING_TENSORS,
    }
)

PARAMS["default"].update(
    {
        "input_channels": [5, 10, 29, 64],
        "outputs": [2, 10, 16, 100],
        "non_matching_tensors": _NON_MATCHING_TENSORS[::2],
    }
)

PARAMS["smoke"].update(
    {
        "input_channels": [5, 29],
        "outputs": [2, 10],
        "non_matching_tensors": _NON_MATCHING_TENSORS[::4],
    }
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def model(input_shape, outputs):
    return build_fc(input_shape=input_shape, outputs=outputs)
