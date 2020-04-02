# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.xcore_model import TensorType
from tflite2xcore.transformation_passes import LegalizeOutputTensorNamePass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_relu, build_consecutive_pads, build_split
)

from .conftest import (
    PARAMS,
    _test_matching_params,
    _test_non_matching_params
)

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update({
    "num_splits": [2, 4]
})

PARAMS["smoke"].update({
    "num_splits": [2]
})


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


@pytest.fixture()
def model_multi_out(input_shape, num_splits):
    return build_split(input_shape=input_shape, num_splits=num_splits,
                       tensor_type=TensorType.INT8, axis=2)


@pytest.fixture()
def model_multi_out_partial(input_shape, num_splits):
    model = build_split(input_shape=input_shape, num_splits=num_splits,
                        tensor_type=TensorType.INT8, axis=2)
    op = model.subgraphs[0].operators[0]
    for j, tensor in enumerate(op.outputs[1:]):
        tensor.name = f"{op.name}/output_{j+1}"
    return model


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_matching_simple(trf_pass, model_simple):
    _test_matching_params(trf_pass, model_simple)


def test_matching_multi_op(trf_pass, model_multi_op):
    operators = model_multi_op.subgraphs[0].operators
    assert trf_pass.match(operators[0])
    assert trf_pass.match(operators[1])


def test_matching_multi_out(trf_pass, model_multi_out):
    _test_matching_params(trf_pass, model_multi_out)


def test_matching_multi_out_partial(trf_pass, model_multi_out_partial):
    _test_matching_params(trf_pass, model_multi_out_partial)


def test_non_matching_simple(trf_pass, model_simple):
    subgraph = model_simple.subgraphs[0]
    op = subgraph.operators[0]
    t_out = op.outputs[0]
    t_out.name = f"{op.name}/output"
    _test_non_matching_params(trf_pass, model_simple)


def test_non_matching_multi_op(trf_pass, model_multi_op):
    subgraph = model_multi_op.subgraphs[0]
    for op in subgraph.operators:
        assert len(op.outputs) == 1
        t_out = op.outputs[0]
        t_out.name = f"{op.name}/output"

    for j, op in enumerate(subgraph.operators):
        assert not trf_pass.match(op), f"op {j} should not be matched"


def test_non_matching_multi_out(trf_pass, model_multi_out):
    subgraph = model_multi_out.subgraphs[0]
    op = subgraph.operators[0]
    for j, tensor in enumerate(op.outputs):
        tensor.name = f"{op.name}/output_{j}"

    _test_non_matching_params(trf_pass, model_multi_out)


def test_mutate_simple(trf_pass, model_simple):
    # run mutating pass
    trf_pass.run(model_simple)
    model_simple.sanity_check()

    op = model_simple.subgraphs[0].operators[-1]
    assert op.outputs[0].name == f"{op.name}/output"


def test_mutate_multi_op(trf_pass, model_multi_op):
    # run mutating pass
    trf_pass.run(model_multi_op)
    model_multi_op.sanity_check()

    for j, op in enumerate(model_multi_op.subgraphs[0].operators):
        expected_name = f"{op.name}/output"
        name = op.outputs[0].name
        assert name == expected_name, f"op {j} name: expected '{expected_name}', found '{name}'"


def test_mutate_multi_out(trf_pass, model_multi_out):
    # run mutating pass
    trf_pass.run(model_multi_out)
    model_multi_out.sanity_check()

    op = model_multi_out.subgraphs[0].operators[-1]
    for j, tensor in enumerate(op.outputs):
        expected_name = f"{op.name}/output_{j}"
        name = tensor.name
        assert name == expected_name, f"tensor {j} name: expected '{expected_name}', found '{name}'"


def test_mutate_multi_out_partial(trf_pass, model_multi_out_partial):
    # run mutating pass
    trf_pass.run(model_multi_out_partial)
    model_multi_out_partial.sanity_check()

    op = model_multi_out_partial.subgraphs[0].operators[-1]
    for j, tensor in enumerate(op.outputs):
        expected_name = f"{op.name}/output_{j}"
        name = tensor.name
        assert name == expected_name, f"tensor {j} name: expected '{expected_name}', found '{name}'"


if __name__ == "__main__":
    pytest.main()