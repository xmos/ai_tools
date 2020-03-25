# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = {
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
def input_size(input_height, input_width):
    return [input_height, input_width]


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
