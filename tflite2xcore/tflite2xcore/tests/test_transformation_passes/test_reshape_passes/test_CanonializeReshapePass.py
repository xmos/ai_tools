# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from copy import deepcopy

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes.reshape_passes import CanonicalizeReshapePass
from tflite2xcore.xcore_schema import BuiltinOpCodes

from ..model_builders import build_fc_with_reshape, build_reshape
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    update_params_with_reshape,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------


def matching_reshape(reshape_input_shape, reshape_output_shape):
    return True


PARAMS = update_params_with_reshape(deepcopy(PARAMS), is_matching=matching_reshape)

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return CanonicalizeReshapePass()


@pytest.fixture()
def model(reshape):
    return build_reshape(input_shape=reshape["input"], output_shape=reshape["output"])


@pytest.fixture()
def model_no_shape_tensor(reshape):
    return build_reshape(
        input_shape=reshape["input"],
        output_shape=reshape["output"],
        input_shape_tensor=False,
    )


def test_mutate(trf_pass, model):

    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 1

    in_ori, out_ori = subgraph.inputs[0], subgraph.outputs[0]

    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    # need to clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    assert len(subgraph.operators) == 1
    op = subgraph.operators[0]
    assert len(op.inputs) == 1
    assert len(op.outputs) == 1

    # check input/output tensors
    assert len(subgraph.inputs) == len(subgraph.outputs) == 1
    in_new, out_new = subgraph.inputs[0], subgraph.outputs[0]
    assert in_new is in_ori is op.inputs[0]
    assert out_ori is out_new is op.outputs[0]


def test_non_matching_no_shape_tensor(trf_pass, model_no_shape_tensor):
    _test_non_matching_params(trf_pass, model_no_shape_tensor)


if __name__ == "__main__":
    pytest.main()
