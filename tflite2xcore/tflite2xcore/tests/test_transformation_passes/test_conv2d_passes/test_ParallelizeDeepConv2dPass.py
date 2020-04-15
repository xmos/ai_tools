# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import ParallelizeDeepConv2dPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_XC_conv2d_deep
from .test_ReplaceDeepConv2dPass import PARAMS


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update({
    "num_threads": [2, 3, 4, 5]
})

PARAMS["smoke"].update({
    "num_threads": [2, 5]
})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def trf_pass(num_threads):
    return ParallelizeDeepConv2dPass(num_threads=num_threads)


@pytest.fixture()
def model(weight_shape, input_size, strides):
    return build_XC_conv2d_deep(weight_shape=weight_shape,
                                input_size=input_size,
                                strides=strides)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_matching(trf_pass, model, num_threads):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


def test_single_thread_identity(model):
    trf_pass = ParallelizeDeepConv2dPass(num_threads=1)
    op = model.subgraphs[0].operators[0]
    custom_options = deepcopy(op.custom_options)
    trf_pass.run(model)
    model.sanity_check()
    assert op.custom_options == custom_options


def test_mutate(trf_pass, model, num_threads):
    op = model.subgraphs[0].operators[0]
    assert 'par_plan' not in op.custom_options
    trf_pass.run(model)
    model.sanity_check()
    assert 'par_plan' in op.custom_options


if __name__ == "__main__":
    pytest.main()
