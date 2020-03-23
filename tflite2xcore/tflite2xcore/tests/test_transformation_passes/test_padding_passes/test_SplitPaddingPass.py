# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import SplitPaddingPass

from ..model_builders import build_pad
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PADS = [0, 1, 2]

PARAMS["default"].update({
    "input_batches": [1, 4],
    "pad_t": PADS,
    "pad_b": PADS,
    "pad_l": PADS,
    "pad_r": PADS
})

PARAMS["smoke"].update({
    "input_batches": [1],
    "pad_t": PADS[:2],
    "pad_b": PADS[:2],
    "pad_l": PADS[:2],
    "pad_r": PADS[:2]
})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def build_model():
    return build_pad


@pytest.fixture()
def trf_pass():
    return SplitPaddingPass()


@pytest.fixture()
def input_shape(input_batches, input_size, input_channels):
    return [input_batches, *input_size, input_channels]


@pytest.fixture()
def paddings_HW(pad_t, pad_b, pad_l, pad_r):
    return [(0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)]


@pytest.fixture()
def paddings_NC(pad_t, pad_b, pad_l, pad_r):
    return [(pad_t, pad_b), (0, 0), (0, 0), (pad_l, pad_r)]


@pytest.fixture()
def paddings(paddings_HW, paddings_NC):
    pads = [paddings_NC[0], *paddings_HW[1:3], paddings_NC[3]]
    if sum(sum(p) for p in pads) == 0:
        pytest.skip("skipping constant zero padding case")
    return pads


@pytest.fixture()
def model(input_shape, paddings):
    return build_pad(input_shape=input_shape, paddings=paddings)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_non_matching_HW_only(trf_pass, build_model, input_shape, paddings_HW):
    model = build_pad(input_shape=input_shape, paddings=paddings_HW)
    _test_non_matching_params(trf_pass, model)


def test_non_matching_NC_only(trf_pass, build_model, input_shape, paddings_NC):
    model = build_pad(input_shape=input_shape, paddings=paddings_NC)
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
