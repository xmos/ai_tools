# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import SplitPaddingPass
from tflite2xcore.operator_codes import BuiltinOpCodes

from ..model_builders import build_pad
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PADS = [0, 1, 2]

PARAMS["default"].update({
    "input_batches": [1, 4],
    "pad_t": PADS,
    "pad_b": PADS,
    "pad_l": PADS,
    "pad_r": PADS
})

PARAMS["smoke"].update({
    "input_batches": [1],
    "pad_t": PADS[:2],
    "pad_b": PADS[:2],
    "pad_l": PADS[:2],
    "pad_r": PADS[:2]
})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def build_model():
    return build_pad


@pytest.fixture()
def trf_pass():
    return SplitPaddingPass()


@pytest.fixture()
def input_shape(input_batches, input_size, input_channels):
    return [input_batches, *input_size, input_channels]


@pytest.fixture()
def paddings_HW(pad_t, pad_b, pad_l, pad_r):
    return [(0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)]


@pytest.fixture()
def paddings_NC(pad_t, pad_b, pad_l, pad_r):
    return [(pad_t, pad_b), (0, 0), (0, 0), (pad_l, pad_r)]


@pytest.fixture()
def paddings(paddings_HW, paddings_NC):
    pads = [paddings_NC[0], *paddings_HW[1:3], paddings_NC[3]]
    if sum(sum(p) for p in pads) == 0:
        pytest.skip("skipping constant zero padding case")
    return pads


@pytest.fixture()
def model(input_shape, paddings):
    return build_pad(input_shape=input_shape, paddings=paddings)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_mutate(trf_pass, model):
    # extract original tensor shapes:
    subgraph = model.subgraphs[0]
    params_ori = subgraph.operators[-1].inputs[1].numpy.tolist()

    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    # check operators
    operators = subgraph.operators
    assert len(operators) == 2
    op_NC, op_HW = operators
    assert op_NC.operator_code.code is op_HW.operator_code.code is BuiltinOpCodes.PAD
    assert len(op_NC.inputs) == len(op_HW.inputs) == 2
    assert len(op_NC.outputs) == len(op_HW.outputs) == 1

    # check input/output tensors
    assert len(subgraph.inputs) == len(subgraph.outputs) == 1
    input_tensor, output_tensor = subgraph.inputs[0], subgraph.outputs[0]
    assert input_tensor in op_NC.inputs
    assert input_tensor not in op_NC.outputs + op_HW.inputs + op_HW.outputs
    assert output_tensor in op_HW.outputs
    assert output_tensor not in op_HW.inputs + op_NC.inputs + op_NC.outputs

    # check wiring
    assert op_NC.outputs[0] is op_HW.inputs[0]

    # check parameters
    params_NC = op_NC.inputs[1].numpy.tolist()
    params_HW = op_HW.inputs[1].numpy.tolist()
    assert params_NC[1] == params_NC[2] == [0, 0]
    assert params_HW[0] == params_HW[3] == [0, 0]
    assert params_NC[0] == params_ori[0]
    assert params_NC[3] == params_ori[3]
    assert params_HW[1] == params_ori[1]
    assert params_HW[2] == params_ori[2]


def test_non_matching_HW_only(trf_pass, build_model, input_shape, paddings_HW):
    model = build_pad(input_shape=input_shape, paddings=paddings_HW)
    _test_non_matching_params(trf_pass, model)


def test_non_matching_NC_only(trf_pass, build_model, input_shape, paddings_NC):
    model = build_pad(input_shape=input_shape, paddings=paddings_NC)
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
