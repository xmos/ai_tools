# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy
from itertools import product

from tflite2xcore.transformation_passes import FuseConv2dPaddingPass

from ..model_builders import build_padded_DW
from ..test_conv2d_passes.conftest import (
    PARAMS, strides,
    _test_non_matching_params,
    test_matching_params
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PADS = [0, 1, 2]

PARAMS["default"].update({
    "pad_t": PADS,
    "pad_b": PADS,
    "pad_l": PADS,
    "pad_r": PADS,
    "non_matching_pad": [p for p in product(PADS, PADS) if p != (0, 0)]
})

PARAMS["default"].update({
    "pad_t": PADS[:2],
    "pad_b": PADS[:2],
    "pad_l": PADS[:2],
    "pad_r": PADS[:2],
    "non_matching_pad": [p for p in product(PADS[:2], PADS[:2]) if p != (0, 0)]
})

PARAMS["smoke"].update({
    "pad_t": PADS[:1],
    "pad_b": PADS[:1],
    "pad_l": PADS[:1],
    "pad_r": PADS[:1],
    "non_matching_pad": [(0, 1)]
})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def build_model():
    return build_padded_DW


@pytest.fixture()
def trf_pass():
    return FuseConv2dPaddingPass()


@pytest.fixture()
def weight_shape(kernel_height, kernel_width, input_channels):
    return [kernel_height, kernel_width, input_channels]


@pytest.fixture()
def paddings(pad_t, pad_b, pad_l, pad_r):
    return [(0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)]


@pytest.fixture()
def model(weight_shape, input_size, paddings, strides):
    return build_padded_DW(weight_shape=weight_shape, input_size=input_size,
                           paddings=paddings, strides=strides)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_non_matching_batch_pad(trf_pass, build_model,
                                weight_shape, input_size, paddings, strides,
                                non_matching_pad):
    paddings[0] = non_matching_pad
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        paddings=paddings, strides=strides)
    _test_non_matching_params(trf_pass, model)


def test_non_matching_channel_pad(trf_pass, build_model,
                                  weight_shape, input_size, paddings, strides,
                                  non_matching_pad):
    paddings[3] = non_matching_pad
    model = build_model(weight_shape=weight_shape, input_size=input_size,
                        paddings=paddings, strides=strides)
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
