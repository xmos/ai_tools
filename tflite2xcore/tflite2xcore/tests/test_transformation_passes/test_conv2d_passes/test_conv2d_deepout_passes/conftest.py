# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from ..conftest import (
    PARAMS,
    _test_non_matching_dim,
    test_matching_params,
    test_non_matching_output_channels,
    test_non_matching_kernel_height,
    test_non_matching_kernel_width,
    test_non_matching_input_channels,
    test_non_matching_types
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update({
    "output_channels": [16, 32],
    "non_matching_output_channels": [8, 24, 17, 63],
    "kernel_height": [1, 3, 5, 7],
    "non_matching_kernel_height": [2, 4, 6],
    "kernel_width": [1, 3, 5, 7]
})

PARAMS["smoke"].update({
    "output_channels": [16],
    "non_matching_output_channels": [8],
    "kernel_height": [1, 3],
    "non_matching_kernel_height": [2],
    "kernel_width": [1, 3]
})


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_non_matching_stride_w(trf_pass, model, non_matching_stride_w):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_w'] = non_matching_stride_w
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


def test_non_matching_stride_h(trf_pass, model, non_matching_stride_h):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_h'] = non_matching_stride_h
    assert not trf_pass.match(model.subgraphs[0].operators[-1])
