# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes import RemovePaddingInputPass
from tflite2xcore.xcore_schema import BuiltinOpCodes

from ..model_builders import build_pad, build_non_input_pad
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    update_params_with_paddings,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------


def only_channel_pad(padding):
    return padding[-1] != [0, 0] and (all(pad == [0, 0] for pad in padding[:-1]))


PARAMS = update_params_with_paddings(deepcopy(PARAMS), is_matching=only_channel_pad)

# NOTE: this is intentional to keep test case count lower
PARAMS["default"]["paddings"] = PARAMS["smoke"]["paddings"]


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return RemovePaddingInputPass()


@pytest.fixture()
def model(input_shape, paddings):
    return build_pad(input_shape=input_shape, paddings=paddings)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(trf_pass, model):
    # extract original padding values
    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 1
    pad_ori = subgraph.operators[0].inputs[1]
    paddings_ori = pad_ori.numpy.tolist()
    in_ori, out_ori = subgraph.inputs[0], subgraph.outputs[0]

    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    # need to clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    # check pad operator has been removed
    assert len(subgraph.operators) == 0

    # check input/output tensors - new input/output should be old output
    assert len(subgraph.inputs) == len(subgraph.outputs) == 1
    in_new, out_new = subgraph.inputs[0], subgraph.outputs[0]
    assert in_new is out_ori
    assert out_ori is out_new


def test_non_matching_non_input_pad(trf_pass, input_shape, paddings):
    model = build_non_input_pad(input_shape=input_shape, paddings=paddings)

    for op in model.subgraphs[0].operators:
        assert not trf_pass.match(op)


if __name__ == "__main__":
    pytest.main()
