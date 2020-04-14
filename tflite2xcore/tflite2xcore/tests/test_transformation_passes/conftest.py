# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = {
    "extended": {
        "input_height": [7, 9, 17, 20, 32],
        "input_width": [7, 9, 17, 20, 32],
        "input_channels": [4, 8, 16, 32, 36, 64]
    },
    "default": {
        "input_height": [9, 20],
        "input_width": [7, 17],
        "input_channels": [4, 8, 16, 32]
    },
    "smoke": {
        "input_height": [9, 20],
        "input_width": [7, 17],
        "input_channels": [4, 32]
    }
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def strides(stride_h, stride_w):
    return (stride_h, stride_w)


@pytest.fixture()
def input_size(input_height, input_width):
    return [input_height, input_width]


@pytest.fixture()
def input_shape(input_size, input_channels):
    return [*input_size, input_channels]


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------

def _test_non_matching_params(trf_pass, model):
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


def _test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------

def test_matching_params(trf_pass, model):
    _test_matching_params(trf_pass, model)


def test_non_matching_tensors(trf_pass, model, non_matching_tensors):
    subgraph = model.subgraphs[0]
    subgraph.get_tensor(non_matching_tensors[0]).type = non_matching_tensors[1]
    _test_non_matching_params(trf_pass, model)
