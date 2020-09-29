# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import numpy as np
from copy import deepcopy
from itertools import product, chain
from typing import Tuple, NamedTuple, Callable

from ..conftest import (
    ParamsType,
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {"input_batch": [1, 2], "input_channels": [4, 8, 16], "outputs": [2, 10],}
)

PARAMS["default"].update(
    {"input_batch": [1, 2], "input_channels": [4, 32], "outputs": [2, 10],}
)

PARAMS["smoke"].update(
    {"input_batch": [1], "input_channels": [4, 32], "outputs": [10],}
)


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------


class ReshapeTuple(NamedTuple):
    input: Tuple[int, ...]
    output: Tuple[int, ...]


def update_params_with_reshape(
    PARAMS: ParamsType, *, is_matching: Callable[[ReshapeTuple], bool]
) -> ParamsType:
    for params in PARAMS.values():

        def get_product_shape(*, dim=4, order="NHWC"):
            if dim == 4:
                if order == "NHWC":
                    return product(
                        params["input_batch"],
                        params["input_height"],
                        params["input_width"],
                        params["input_channels"],
                    )
                else:
                    return product(
                        params["input_batch"],
                        params["input_channels"],
                        params["input_height"],
                        params["input_width"],
                    )
            else:
                return (
                    (*p[: dim - 1], np.prod(p[dim - 1 :]))
                    for p in get_product_shape(dim=dim + 1, order=order)
                )

        all_reshapes = (
            ReshapeTuple(*p)
            for p in chain(
                product(
                    get_product_shape(dim=4), get_product_shape(dim=4, order="NCHW")
                ),
                product(get_product_shape(dim=4), get_product_shape(dim=3)),
                product(get_product_shape(dim=3), get_product_shape(dim=2)),
            )
        )

        matching_reshape = params["reshape"] = []
        non_matching_reshape = params["non_matching_reshape"] = []
        for reshape in all_reshapes:
            # this is a bit wasteful
            if np.prod(reshape.input) == np.prod(reshape.output):
                if is_matching(reshape):
                    matching_reshape.append(reshape)
                else:
                    non_matching_reshape.append(reshape)

    return PARAMS
