# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from ..conftest import (
    _pytest_generate_tests,
    _test_non_matching_output_channels,
    _test_non_matching_kernel_height,
    _test_non_matching_kernel_width,
    _test_non_matching_input_channels,
    _test_non_matching_types
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = {
    "default": {
        "output_channels": [16, 32],
        "non_matching_output_channels": [8, 24, 17, 63],
        "kernel_height": [1, 3, 5, 7],
        "non_matching_kernel_height": [2, 4, 6],
        "kernel_width": [1, 3, 5, 7],
        "strides": [(1, 1)],
        "non_matching_stride_w": [2, 3],
        "non_matching_stride_h": [2, 3]
    },
    "smoke": {
        "output_channels": [16],
        "non_matching_output_channels": [8],
        "kernel_height": [1, 3],
        "non_matching_kernel_height": [2],
        "kernel_width": [1, 3],
        "strides": [(1, 1)],
        "non_matching_stride_w": [2],
        "non_matching_stride_h": [2]
    }
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    _pytest_generate_tests(metafunc, PARAMS)


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------

def _test_non_matching_stride_w(trf_pass, model, non_matching_stride_w):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_w'] = non_matching_stride_w
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


def _test_non_matching_stride_h(trf_pass, model, non_matching_stride_h):
    op = model.subgraphs[0].operators[0]
    op.builtin_options['stride_h'] = non_matching_stride_h
    assert not trf_pass.match(model.subgraphs[0].operators[-1])
