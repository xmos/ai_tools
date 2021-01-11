# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes import FuseConsecutivePadsPass

from ..model_builders import build_pad, build_consecutive_pads
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    update_params_with_paddings,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = update_params_with_paddings(deepcopy(PARAMS), is_matching=lambda padding: True)

# NOTE: this is intentional to keep test case count lower
PARAMS["default"]["paddings"] = PARAMS["smoke"]["paddings"]


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model():
    return build_pad


@pytest.fixture()
def trf_pass():
    return FuseConsecutivePadsPass()


@pytest.fixture()
def model(input_shape, paddings, paddings_NC):
    return build_consecutive_pads(
        input_shape=input_shape, paddings_1=paddings, paddings_2=paddings_NC
    )


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(trf_pass, model):
    # extract original padding values
    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 2
    pad_1_ori = subgraph.operators[0].inputs[1]
    pad_2_ori = subgraph.operators[1].inputs[1]
    paddings_1_ori = pad_1_ori.as_array()
    paddings_2_ori = pad_2_ori.as_array()
    in_ori, out_ori = subgraph.inputs[0], subgraph.outputs[0]

    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    # need to clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    # check operator
    assert len(subgraph.operators) == 1
    op = subgraph.operators[0]
    assert len(op.inputs) == 2
    assert len(op.outputs) == 1

    # check input/output tensors
    assert len(subgraph.inputs) == len(subgraph.outputs) == 1
    in_new, out_new = subgraph.inputs[0], subgraph.outputs[0]
    assert in_ori is in_new is op.inputs[0]
    assert out_ori is out_new is op.outputs[0]

    # check parameters
    pad_new = subgraph.operators[0].inputs[1]
    paddings_new = pad_new.as_array()
    assert pad_new is not pad_1_ori
    assert pad_new is not pad_2_ori
    assert paddings_new[0][0] == paddings_1_ori[0][0] + paddings_2_ori[0][0]
    assert paddings_new[0][1] == paddings_1_ori[0][1] + paddings_2_ori[0][1]
    assert paddings_new[1][0] == paddings_1_ori[1][0] + paddings_2_ori[1][0]
    assert paddings_new[1][1] == paddings_1_ori[1][1] + paddings_2_ori[1][1]
    assert paddings_new[2][0] == paddings_1_ori[2][0] + paddings_2_ori[2][0]
    assert paddings_new[2][1] == paddings_1_ori[2][1] + paddings_2_ori[2][1]
    assert paddings_new[3][0] == paddings_1_ori[3][0] + paddings_2_ori[3][0]
    assert paddings_new[3][1] == paddings_1_ori[3][1] + paddings_2_ori[3][1]


def test_non_matching_single_pad(trf_pass, input_shape, paddings):
    model = build_pad(input_shape=input_shape, paddings=paddings)
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
