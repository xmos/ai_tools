# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import SplitPaddingPass
from tflite2xcore.xcore_schema import BuiltinOpCodes

from ..model_builders import build_pad
from . import test_non_matching_paddings
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    update_params_with_paddings,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------


def is_matching(padding):
    return (padding[0] != (0, 0) or padding[3] != (0, 0)) and (
        padding[1] != (0, 0) or padding[2] != (0, 0)
    )


PARAMS = update_params_with_paddings(deepcopy(PARAMS), is_matching=is_matching)


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
def model(input_shape, paddings):
    return build_pad(input_shape=input_shape, paddings=paddings)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(trf_pass, model):
    # extract original padding values
    subgraph = model.subgraphs[0]
    params_ori = subgraph.operators[-1].inputs[1].as_array()

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
    params_NC = op_NC.inputs[1].as_array()
    assert params_NC[1][0] == params_NC[2][0] == 0
    assert params_NC[1][1] == params_NC[2][1] == 0
    assert params_NC[0][0] == params_ori[0][0]
    assert params_NC[0][1] == params_ori[0][1]
    assert params_NC[3][0] == params_ori[3][0]
    assert params_NC[3][1] == params_ori[3][1]

    params_HW = op_HW.inputs[1].as_array()
    assert params_HW[0][0] == params_HW[3][0] == 0
    assert params_HW[0][1] == params_HW[3][1] == 0
    assert params_HW[1][0] == params_ori[1][0]
    assert params_HW[1][1] == params_ori[1][1]
    assert params_HW[2][0] == params_ori[2][0]
    assert params_HW[2][1] == params_ori[2][1]


if __name__ == "__main__":
    pytest.main()
