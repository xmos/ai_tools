# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.xcore_schema import Padding

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
        "kernel_height": [2, 3, 5, 7],
        "kernel_width": [2, 3, 5, 7],
        "input_channels": [256, 512],
        "non_matching_input_channels": [64, 128, 500],
        "output_channels": [32, 64],
        "non_matching_output_channels": [16, 48, 52],
        "padding": list(Padding),
        "stride_h": [1],
        "stride_w": [1],
        "non_matching_stride_h": [2, 3],
        "non_matching_stride_w": [2, 3],
        "non_matching_dilation_h_factor": [2],
        "non_matching_dilation_w_factor": [2],
    }
)

PARAMS["default"].update(
    {
        "kernel_height": [2, 3, 5],
        "kernel_width": [2, 3, 5],
        "input_channels": [256, 512],
        "non_matching_input_channels": [3, 15],
        "output_channels": [32, 64],
        "non_matching_output_channels": [16, 15],
        "padding": list(Padding),
        "stride_h": [1],
        "stride_w": [1],
        "non_matching_stride_h": [2, 3],
        "non_matching_stride_w": [2, 3],
        "non_matching_dilation_h_factor": [2],
        "non_matching_dilation_w_factor": [2],
    }
)

PARAMS["smoke"].update(
    {
        "kernel_height": [2, 3],
        "kernel_width": [2, 3],
        "input_channels": [256],
        "non_matching_input_channels": [128],
        "output_channels": [32],
        "non_matching_output_channels": [16],
        "padding": list(Padding),
        "stride_h": [1],
        "stride_w": [1],
        "non_matching_stride_h": [2],
        "non_matching_stride_w": [2],
        "non_matching_dilation_h_factor": [2],
        "non_matching_dilation_w_factor": [2],
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


def test_non_matching_dilation_w_factor(trf_pass, model, non_matching_dilation_w_factor):
    op = model.subgraphs[0].operators[0]
    op.builtin_options["dilation_w_factor"] = non_matching_dilation_w_factor
    _test_non_matching_params(trf_pass, model)


def test_non_matching_dilation_h_factor(trf_pass, model, non_matching_dilation_h_factor):
    op = model.subgraphs[0].operators[0]
    op.builtin_options["dilation_h_factor"] = non_matching_dilation_h_factor
    _test_non_matching_params(trf_pass, model)
