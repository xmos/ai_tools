# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

MATCHING_OUTPUT_CHANNELS = [16, 32]
MATCHING_KERNEL_HEIGHT = [1, 3, 5, 7]
MATCHING_KERNEL_WIDTH = MATCHING_KERNEL_HEIGHT
MATCHING_STRIDES = (1, 1)

NON_MATCHING_STRIDE_W = [2, 3]
NON_MATCHING_STRIDE_H = NON_MATCHING_STRIDE_W
NON_MATCHING_OUTPUT_CHANNELS = [8, 24, 17, 63]
NON_MATCHING_KERNEL_HEIGHT = [2, 4, 6]


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def strides():
    return MATCHING_STRIDES


@pytest.fixture(params=MATCHING_OUTPUT_CHANNELS)
def output_channels(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_OUTPUT_CHANNELS)
def non_matching_output_channels(request):
    return request.param


@pytest.fixture(params=MATCHING_KERNEL_HEIGHT)
def kernel_height(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_KERNEL_HEIGHT)
def non_matching_kernel_height(request):
    return request.param


@pytest.fixture(params=MATCHING_KERNEL_WIDTH)
def kernel_width(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_STRIDE_W)
def non_matching_stride_w(request):
    return request.param


@pytest.fixture(params=NON_MATCHING_STRIDE_H)
def non_matching_stride_h(request):
    return request.param
