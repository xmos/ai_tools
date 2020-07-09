# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from copy import deepcopy

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes.reshape_passes import RemoveFlattenReshapePass
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

    # Check batch dim is unchanged
    reshape_input_batch = 1
    reshape_output_batch = 1

    if len(reshape_output_shape) == 4:
        reshape_output_batch = reshape_output_shape[0]

    if len(reshape_input_shape) == 4:
        reshape_input_batch = reshape_input_shape[0]

    return reshape_input_batch == reshape_output_batch


PARAMS = update_params_with_reshape(deepcopy(PARAMS), is_matching=matching_reshape)

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return RemoveFlattenReshapePass()


@pytest.fixture()
def model(outputs, reshape):

    return build_fc_with_reshape(
        input_shape=reshape[0], fc_outputs=outputs, reshaped_input_shape=reshape[1]
    )


@pytest.fixture()
def model_nonmatch(outputs, non_matching_reshape):
    return build_fc_with_reshape(
        input_shape=non_matching_reshape[0],
        fc_outputs=outputs,
        reshaped_input_shape=non_matching_reshape[1],
    )


@pytest.fixture()
def model_reshape_only(outputs, reshape):
    return build_reshape(input_shape=reshape[0], output_shape=reshape[1],)


def test_mutate(trf_pass, model):

    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 2

    in_ori, out_ori = subgraph.inputs[0], subgraph.outputs[0]

    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    # need to clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    # Check FC operator and that RESHAPE has been removed
    assert len(subgraph.operators) == 1
    op = subgraph.operators[0]
    assert len(op.inputs) == 3
    assert len(op.outputs) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.FULLY_CONNECTED

    # check input/output tensors
    assert len(subgraph.inputs) == len(subgraph.outputs) == 1
    in_new, out_new = subgraph.inputs[0], subgraph.outputs[0]
    assert in_new is in_ori is op.inputs[0]
    assert out_ori is out_new is op.outputs[0]


def test_non_matching_reshape_only(trf_pass, model_reshape_only):
    _test_non_matching_params(trf_pass, model_reshape_only)


def test_non_matching_simple(trf_pass, model_nonmatch):
    _test_non_matching_params(trf_pass, model_nonmatch)


if __name__ == "__main__":
    pytest.main()
