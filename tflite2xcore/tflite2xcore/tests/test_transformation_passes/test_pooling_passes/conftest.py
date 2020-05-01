# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from ..conftest import PARAMS, _test_non_matching_params, test_matching_params


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update(
    {
        "non_matching_input_channels": [1, 3, 9, 15],
        "padding": ["VALID"],
        "non_matching_padding": ["SAME"],
        "fused_activation": ["NONE"],
        "non_matching_fused_activation": ["RELU", "RELU6"],
        "stride_h": [1, 2],
        "stride_w": [1, 2],
        "pool_h": [1, 2, 3],
        "pool_w": [1, 2, 3],
    }
)

PARAMS["smoke"].update(
    {
        "non_matching_input_channels": [1, 9],
        "padding": ["VALID"],
        "non_matching_padding": ["SAME"],
        "fused_activation": ["NONE"],
        "non_matching_fused_activation": ["RELU"],
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
