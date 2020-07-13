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
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {"input_batch": [1, 2], "input_channels": [4, 8, 16, 32, 64], "outputs": [2, 10],}
)

PARAMS["default"].update(
    {"input_batch": [1, 2], "input_channels": [4, 32], "outputs": [2, 10],}
)

PARAMS["smoke"].update(
    {"input_batch": [1], "input_channels": [4, 32], "outputs": [10],}
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

# TODO refactor this function to use a function that returns a generator
def update_params_with_reshape(PARAMS, *, is_matching):

    for params in PARAMS.values():

        assert len(params["input_channels"]) == len(params["input_width"])

        all_reshapes = [
            [list(p) for p in t]
            for t in product(
                product(
                    params["input_batch"],
                    params["input_height"],
                    params["input_width"],
                    params["input_channels"],
                ),
                product(
                    # Basic dim re-ordering
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
                    params["input_height"],
                    params["input_width"],
                    params["input_channels"],
                ),
                # Basic dimensionality reduction
                product(
                    params["input_batch"],
                    params["input_height"],
                    (
                        np.array(params["input_width"])
                        * np.array(params["input_channels"])
                    ).tolist(),
                ),
            )
        )

        all_reshapes.extend(
            [list(p) for p in t]
            for t in product(
                product(
                    params["input_batch"],
                    params["input_height"],
                    params["input_width"],
                ),
                product(
                    params["input_batch"],
                    (
                        np.array(params["input_height"])
                        * np.array(params["input_width"])
                    ).tolist(),
                ),
            )
        )

        for x in range(0, len(all_reshapes)):
            all_reshapes[x] = {
                "input": all_reshapes[x][0],
                "output": all_reshapes[x][1],
            }

        params.update(
            {
                "reshape": [
                    reshape
                    for reshape in all_reshapes
                    # Note, this is a bit of a waste as we collect lots of params them throw a lot of them away..
                    if is_matching(reshape["input"], reshape["output"])
                    and np.prod(reshape["input"]) == np.prod(reshape["output"])
                ],
                "non_matching_reshape": [
                    reshape
                    for reshape in all_reshapes
                    # Note, this is a bit of a waste as we collect lots of params them throw a lot of them away..
                    if (not is_matching(reshape["input"], reshape["output"]))
                    and np.prod(reshape["input"]) == np.prod(reshape["output"])
                ],
            }
        )

    return PARAMS
