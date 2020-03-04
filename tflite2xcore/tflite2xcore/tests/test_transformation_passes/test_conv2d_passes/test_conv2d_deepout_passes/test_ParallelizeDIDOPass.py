# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import ParallelizeDIDOPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_DIDO
from . import test_ReplaceDeepinDeepoutConv2DPass
from .conftest import PARAMS


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update({
    "input_channels": test_ReplaceDeepinDeepoutConv2DPass.PARAMS["default"]["input_channels"]
})

PARAMS["smoke"].update({
    "input_channels": test_ReplaceDeepinDeepoutConv2DPass.PARAMS["smoke"]["input_channels"]
})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def trf_pass(num_threads):
    return ParallelizeDIDOPass(num_threads=num_threads)


@pytest.fixture()
def model(weight_shape, input_size, padding):
    return build_DIDO(weight_shape=weight_shape, input_size=input_size, padding=padding)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

@pytest.mark.parametrize('num_threads', [2])
def test_matching(trf_pass, model, num_threads):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('num_threads', [1])
def test_single_thread_identity(trf_pass, model, num_threads):
    op = model.subgraphs[0].operators[0]
    custom_options = deepcopy(op.custom_options)
    trf_pass.run(model)
    model.sanity_check()
    assert op.custom_options == custom_options


@pytest.mark.parametrize('num_threads', [2, 3, 4, 5])
def test_mutate(trf_pass, model, num_threads):
    op = model.subgraphs[0].operators[0]
    assert 'par_plan' not in op.custom_options
    trf_pass.run(model)
    model.sanity_check()
    assert 'par_plan' in op.custom_options


if __name__ == "__main__":
    pytest.main()
