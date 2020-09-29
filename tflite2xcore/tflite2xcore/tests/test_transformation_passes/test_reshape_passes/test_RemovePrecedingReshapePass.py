# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes.reshape_passes import RemovePrecedingReshapePass
from tflite2xcore.xcore_schema import BuiltinOpCodes

from ..model_builders import build_fc_with_preceding_reshape, build_reshape
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
    return reshape_input_shape[0] == reshape_output_shape[0]


PARAMS = update_params_with_reshape(deepcopy(PARAMS), is_matching=matching_reshape)

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return RemovePrecedingReshapePass()


@pytest.fixture()
def model(outputs, reshape):
    return build_fc_with_preceding_reshape(
        input_shape=reshape["input"],
        fc_outputs=outputs,
        reshaped_input_shape=reshape["output"],
    )


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


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
    assert op.operator_code.code is BuiltinOpCodes.FULLY_CONNECTED

    # check input/output tensors
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1

    assert in_ori is op.inputs[0]
    assert in_ori in subgraph.inputs
    assert out_ori is op.outputs[0]
    assert out_ori in subgraph.outputs


def test_non_matching_reshape_only(trf_pass, reshape):
    model = build_reshape(input_shape=reshape["input"], output_shape=reshape["output"])
    _test_non_matching_params(trf_pass, model)


def test_non_matching_simple(trf_pass, outputs, non_matching_reshape):
    model = build_fc_with_preceding_reshape(
        input_shape=non_matching_reshape["input"],
        fc_outputs=outputs,
        reshaped_input_shape=non_matching_reshape["output"],
    )
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
