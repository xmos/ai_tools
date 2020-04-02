# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import TensorType
from tflite2xcore.transformation_passes import LegalizeOutputTensorNamePass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_relu, build_consecutive_pads
)

from ..conftest import (
    PARAMS,
    _test_matching_params,
    _test_non_matching_params
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def trf_pass():
    return LegalizeOutputTensorNamePass()


@pytest.fixture()
def model_simple(input_shape):
    return build_relu(input_shape=input_shape, tensor_type=TensorType.INT8)


@pytest.fixture()
def model_multi_op(input_shape):
    paddings = [[0] * 2] * 4
    return build_consecutive_pads(input_shape=[1, *input_shape],
                                  paddings_1=paddings, paddings_2=paddings)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_matching_simple(trf_pass, model_simple):
    _test_matching_params(trf_pass, model_simple)


def test_matching_multi_op(trf_pass, model_multi_op):
    operators = model_multi_op.subgraphs[0].operators
    assert trf_pass.match(operators[0])
    assert trf_pass.match(operators[1])


def test_non_matching_simple(trf_pass, model_simple):
    subgraph = model_simple.subgraphs[0]
    op = subgraph.operators[0]
    t_out = op.outputs[0]
    t_out.name = f"{op.name}/output"
    _test_non_matching_params(trf_pass, model_simple)


def test_non_matching_multi_op(trf_pass, model_multi_op):
    subgraph = model_multi_op.subgraphs[0]
    op0, op1 = subgraph.operators
    t_out_0, t_out_1 = op0.outputs[0], op1.outputs[0]

    t_out_0.name = f"{op0.name}/output"
    t_out_1.name = f"{op1.name}/output"
    assert not trf_pass.match(op0)
    assert not trf_pass.match(op1)


if __name__ == "__main__":
    pytest.main()