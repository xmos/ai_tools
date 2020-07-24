# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.xcore_schema import Padding

from ..test_fully_connected_passes.conftest import PARAMS as FC_PARAMS
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

for key in PARAMS:
    PARAMS[key]["non_matching_tensors"] = FC_PARAMS[key]["non_matching_tensors"]

PARAMS["extended"].update(
    {
        "kernel_height": [2, 3, 5, 7],
        "kernel_width": [2, 3, 5, 7],
        "non_matching_input_channels": [3, 9, 15],
        "output_channels": PARAMS["extended"]["input_channels"],
        "non_matching_output_channels": [3, 9, 15],
        "padding": list(Padding),
        "stride_h": [1, 2, 3],
        "stride_w": [1, 2, 3],
    }
)

PARAMS["default"].update(
    {
        "kernel_height": [2, 3, 5],
        "kernel_width": [2, 3, 5],
        "non_matching_input_channels": [3, 15],
        "input_channels": PARAMS["default"]["input_channels"][1:],
        "output_channels": PARAMS["default"]["input_channels"][1:],
        "non_matching_output_channels": [3, 15],
        "padding": list(Padding),
        "stride_h": [1, 2],
        "stride_w": [1, 2],
    }
)

PARAMS["smoke"].update(
    {
        "kernel_height": [2, 3],
        "kernel_width": [2, 3],
        "non_matching_input_channels": [9],
        "output_channels": PARAMS["smoke"]["input_channels"],
        "non_matching_output_channels": [9],
        "padding": list(Padding),
        "stride_h": [1],
        "stride_w": [1],
    }
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def weight_shape(output_channels, kernel_height, kernel_width, input_channels):
    return [output_channels, kernel_height, kernel_width, input_channels]


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_non_matching_output_channels(
    trf_pass,
    build_model,
    non_matching_output_channels,
    kernel_height,
    kernel_width,
    input_channels,
    input_size,
    padding,
    strides,
):
    model = build_model(
        weight_shape=[
            non_matching_output_channels,
            kernel_height,
            kernel_width,
            input_channels,
        ],
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_kernel_height(
    trf_pass,
    build_model,
    output_channels,
    non_matching_kernel_height,
    kernel_width,
    input_channels,
    input_size,
    padding,
    strides,
):
    model = build_model(
        weight_shape=[
            output_channels,
            non_matching_kernel_height,
            kernel_width,
            input_channels,
        ],
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_kernel_width(
    trf_pass,
    build_model,
    output_channels,
    kernel_height,
    non_matching_kernel_width,
    input_channels,
    input_size,
    padding,
    strides,
):
    model = build_model(
        weight_shape=[
            output_channels,
            kernel_height,
            non_matching_kernel_width,
            input_channels,
        ],
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_input_channels(
    trf_pass,
    build_model,
    output_channels,
    kernel_height,
    kernel_width,
    non_matching_input_channels,
    input_size,
    padding,
    strides,
):
    model = build_model(
        weight_shape=[
            output_channels,
            kernel_height,
            kernel_width,
            non_matching_input_channels,
        ],
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_stride_w(trf_pass, model, non_matching_stride_w):
    op = model.subgraphs[0].operators[0]
    op.builtin_options["stride_w"] = non_matching_stride_w
    _test_non_matching_params(trf_pass, model)


def test_non_matching_stride_h(trf_pass, model, non_matching_stride_h):
    op = model.subgraphs[0].operators[0]
    op.builtin_options["stride_h"] = non_matching_stride_h
    _test_non_matching_params(trf_pass, model)
