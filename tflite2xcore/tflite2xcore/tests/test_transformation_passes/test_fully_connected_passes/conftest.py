# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
from copy import deepcopy
from itertools import product
import numpy as np

from tflite2xcore.xcore_schema import TensorType
from tflite2xcore.transformation_passes import ReplaceFullyConnectedPass

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
    {
        "input_channels": [5, 8, 10, 16, 29, 64],
        "outputs": [1, 2, 10, 16, 29, 100],
        "non_matching_tensors": [
            ("input", TensorType.INT16),
            ("input", TensorType.INT32),
            ("input", TensorType.UINT8),
            ("input", TensorType.FLOAT32),
            ("weights", TensorType.INT16),
            ("weights", TensorType.INT32),
            ("weights", TensorType.UINT8),
            ("weights", TensorType.FLOAT32),
            ("biases", TensorType.INT8),
            ("biases", TensorType.INT16),
            ("biases", TensorType.UINT8),
            ("biases", TensorType.FLOAT32),
            ("output", TensorType.INT16),
            ("output", TensorType.INT32),
            ("output", TensorType.UINT8),
            ("output", TensorType.FLOAT32),
        ],
    }
)

PARAMS["default"].update(
    {
        "input_channels": [5, 10, 29, 64],
        "outputs": [2, 10, 16, 100],
        "non_matching_tensors": [
            ("input", TensorType.INT16),
            ("input", TensorType.INT32),
            ("weights", TensorType.INT16),
            ("weights", TensorType.INT32),
            ("biases", TensorType.INT8),
            ("biases", TensorType.INT16),
            ("output", TensorType.INT16),
            ("output", TensorType.INT32),
        ],
    }
)

PARAMS["smoke"].update(
    {
        "input_channels": [5, 29],
        "outputs": [2, 10],
        "non_matching_tensors": [
            ("input", TensorType.INT16),
            ("weights", TensorType.INT16),
            ("biases", TensorType.INT16),
            ("output", TensorType.INT16),
        ],
    }
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def model(input_shape, outputs):
    return build_fc(input_shape=input_shape, outputs=outputs)


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------


def update_params_with_reshape(PARAMS, *, is_matching):

    random.seed(42)

    for params in PARAMS.values():

        all_reshapes = [
            [list(p) for p in t]
            for t in product(
                product(
                    params["input_channels"],
                    params["input_height"],
                    params["input_width"],
                ),
                product(
                    params["input_channels"],
                    params["input_height"],
                    params["input_width"],
                ),
            )
        ]

        all_reshapes = [
            [list(p) for p in t]
            for t in product(
                product(
                    params["input_channels"],
                    params["input_height"],
                    params["input_width"],
                ),
                product(
                    [1],
                    params["input_channels"],
                    params["input_height"],
                    params["input_width"],
                ),
            )
        ]

        all_reshapes.extend(
            [list(p) for p in t]
            for t in product(
                product(
                    params["input_channels"],
                    params["input_height"],
                    params["input_width"],
                ),
                product(
                    params["input_channels"],
                    (
                        np.array(params["input_height"])
                        * np.array(params["input_width"])
                    ).tolist(),
                ),
            )
        )

        params.update(
            {
                "reshape": [
                    reshape
                    for reshape in all_reshapes
                    if is_matching(reshape[0], reshape[1])
                ],
                "non_matching_reshape": [
                    reshape
                    for reshape in all_reshapes
                    if not is_matching(reshape[0], reshape[1])
                ],
            }
        )

    return PARAMS
