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
        "input_batch": [1, 2],
        "input_channels": [5, 10, 29],
        "outputs": [2, 10],
    }
)

PARAMS["default"].update(
    {
        "input_batch": [1, 2],
        "input_channels": [5, 10, 29],
        "outputs": [2, 10],
    }
)

PARAMS["smoke"].update(
    {
        "input_batch": [1],
        "input_channels": [5, 29],
        "outputs": [10],
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

    for params in PARAMS.values():

        all_reshapes = [
            [list(p) for p in t]
            for t in product(
                product(
                    params["input_batch"],
                    params["input_channels"],
                    params["input_height"],
                    params["input_width"],
                ),
                product(
                    params["input_batch"],
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
                    params["input_batch"],
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
        )

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
                    # Note, this is a bit of a waste as we collect lots of params them throw a lot of them away..
                    if is_matching(reshape[0], reshape[1])
                    and np.prod(reshape[0]) == np.prod(reshape[1])
                ],
                "non_matching_reshape": [
                    reshape
                    for reshape in all_reshapes
                    # Note, this is a bit of a waste as we collect lots of params them throw a lot of them away..
                    if (not is_matching(reshape[0], reshape[1]))
                    and np.prod(reshape[0]) == np.prod(reshape[1])
                ],
            }
        )

    return PARAMS
