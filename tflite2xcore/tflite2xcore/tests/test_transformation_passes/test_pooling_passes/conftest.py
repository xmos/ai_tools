# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.pass_manager import ModelTransformationPass
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import Padding, ActivationFunctionType, XCOREOpCodes

from ..conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    test_replace_mutate as _test_replace_mutate,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update(
    {
        "non_matching_input_channels": [1, 3, 9, 15],
        "padding": [Padding.VALID],
        "non_matching_padding": [Padding.SAME],
        "fused_activation": [ActivationFunctionType.NONE],
        "non_matching_fused_activation": [
            ActivationFunctionType.RELU,
            ActivationFunctionType.RELU6,
        ],
        "stride_h": [1, 2],
        "stride_w": [1, 2],
        "pool_h": [1, 2, 3],
        "pool_w": [1, 2, 3],
    }
)

PARAMS["smoke"].update(
    {
        "non_matching_input_channels": [1, 9],
        "padding": [Padding.VALID],
        "non_matching_padding": [Padding.SAME],
        "fused_activation": [ActivationFunctionType.NONE],
        "non_matching_fused_activation": [ActivationFunctionType.RELU],
        "stride_h": [2],
        "stride_w": [2],
        "pool_h": [2, 3],
        "pool_w": [2, 3],
    }
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def pool_size(pool_h, pool_w):
    return (pool_h, pool_w)


@pytest.fixture()
def model(build_model, input_shape, pool_size, strides, padding, fused_activation):
    return build_model(
        input_shape=input_shape,
        padding=padding,
        pool_size=pool_size,
        strides=strides,
        fused_activation=fused_activation,
    )


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(
    trf_pass: ModelTransformationPass, model: XCOREModel, custom_opcode: XCOREOpCodes
) -> None:
    subgraph = model.subgraphs[0]
    old_op = subgraph.operators[0]
    strides = tuple(old_op.builtin_options[f"stride_{ax}"] for ax in ("h", "w"))
    pool = tuple(old_op.builtin_options[f"filter_{ax}"] for ax in ("height", "width"))

    _test_replace_mutate(trf_pass, model, custom_opcode)

    custom_options = subgraph.operators[-1].custom_options
    assert "pool" in custom_options
    assert custom_options["pool"] == pool
    assert "stride" in custom_options
    assert custom_options["stride"] == strides


def test_non_matching_input_channels(
    trf_pass,
    build_model,
    input_size,
    non_matching_input_channels,
    pool_size,
    strides,
    padding,
    fused_activation,
):
    input_shape = (*input_size, non_matching_input_channels)
    model = build_model(
        input_shape=input_shape,
        padding=padding,
        pool_size=pool_size,
        strides=strides,
        fused_activation=fused_activation,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_fused_activation(
    trf_pass,
    build_model,
    input_shape,
    pool_size,
    strides,
    padding,
    non_matching_fused_activation,
):
    model = build_model(
        input_shape=input_shape,
        padding=padding,
        pool_size=pool_size,
        strides=strides,
        fused_activation=non_matching_fused_activation,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_input_height(
    trf_pass,
    build_model,
    input_width,
    non_matching_input_height,
    input_channels,
    pool_size,
    strides,
    padding,
    fused_activation,
):
    input_shape = (input_width, non_matching_input_height, input_channels)
    model = build_model(
        input_shape=input_shape,
        padding=padding,
        pool_size=pool_size,
        strides=strides,
        fused_activation=fused_activation,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_input_width(
    trf_pass,
    build_model,
    input_width,
    input_height,
    non_matching_input_channels,
    pool_size,
    strides,
    padding,
    fused_activation,
):
    input_shape = (input_width, input_height, non_matching_input_channels)
    model = build_model(
        input_shape=input_shape,
        padding=padding,
        pool_size=pool_size,
        strides=strides,
        fused_activation=fused_activation,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_pool_h(
    trf_pass,
    build_model,
    input_shape,
    non_matching_pool_h,
    pool_w,
    strides,
    padding,
    fused_activation,
):
    model = build_model(
        input_shape=input_shape,
        padding=padding,
        pool_size=(non_matching_pool_h, pool_w),
        strides=strides,
        fused_activation=fused_activation,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_pool_w(
    trf_pass,
    build_model,
    input_shape,
    pool_h,
    non_matching_pool_w,
    strides,
    padding,
    fused_activation,
):
    model = build_model(
        input_shape=input_shape,
        padding=padding,
        pool_size=(pool_h, non_matching_pool_w,),
        strides=strides,
        fused_activation=fused_activation,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_stride_h(
    trf_pass,
    build_model,
    input_shape,
    pool_size,
    padding,
    non_matching_stride_h,
    stride_w,
    fused_activation,
):
    model = build_model(
        input_shape=input_shape,
        padding=padding,
        pool_size=pool_size,
        strides=(non_matching_stride_h, stride_w),
        fused_activation=fused_activation,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_stride_w(
    trf_pass,
    build_model,
    input_shape,
    pool_size,
    padding,
    stride_h,
    non_matching_stride_w,
    fused_activation,
):
    model = build_model(
        input_shape=input_shape,
        padding=padding,
        pool_size=pool_size,
        strides=(stride_h, non_matching_stride_w),
        fused_activation=fused_activation,
    )
    _test_non_matching_params(trf_pass, model)
