# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy
from itertools import product

from ..conftest import (
    PARAMS,
    ParamsType,
    _test_non_matching_params,
    test_matching_params,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PADS = [0, 1, 2]

PARAMS["extended"].update(
    {
        "input_batches": [1, 4],
        "pad_t": PADS,
        "pad_b": PADS,
        "pad_l": PADS,
        "pad_r": PADS,
        "pad_batch_l": [0],
        "pad_batch_r": PADS,
        "pad_channel_l": [0],
        "pad_channel_r": PADS,
    }
)

PARAMS["default"].update(
    {
        "input_batches": [1],
        "pad_t": PADS,
        "pad_b": PADS[:2],
        "pad_l": PADS,
        "pad_r": PADS[:2],
        "pad_batch_l": [0],
        "pad_batch_r": [0],
        "pad_channel_l": [0],
        "pad_channel_r": PADS,
    }
)

PARAMS["smoke"].update(
    {
        "input_batches": [1],
        "pad_t": PADS[:1],
        "pad_b": PADS,
        "pad_l": PADS[:1],
        "pad_r": PADS,
        "pad_batch_l": [0],
        "pad_batch_r": [0],
        "pad_channel_l": [0],
        "pad_channel_r": PADS[:2],
    }
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def input_shape(input_batches, input_size, input_channels):
    return [input_batches, *input_size, input_channels]


@pytest.fixture()
def paddings_HW(pad_t, pad_b, pad_l, pad_r):
    return [(0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)]


@pytest.fixture()
def paddings_NC(pad_batch_l, pad_batch_r, pad_channel_l, pad_channel_r):
    return [(pad_batch_r, pad_batch_r), (0, 0), (0, 0), (pad_channel_l, pad_channel_r)]


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------


def update_params_with_paddings(PARAMS, *, is_matching):
    for params in PARAMS.values():
        all_paddings = list(
            product(
                product(params["pad_batch_l"], params["pad_batch_r"]),
                product(params["pad_t"], params["pad_b"]),
                product(params["pad_l"], params["pad_r"]),
                product(params["pad_channel_l"], params["pad_channel_r"]),
            )
        )

        params.update(
            {
                "paddings": [
                    padding for padding in all_paddings if is_matching(padding)
                ],
                "non_matching_paddings": [
                    padding for padding in all_paddings if not is_matching(padding)
                ],
            }
        )

    return PARAMS
