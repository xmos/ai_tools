# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
import numpy as np

from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes, XCOREOpCodes
from tflite2xcore.transformation_passes import ReplaceArgMax16Pass

from .test_AddArgMax16OutputPass import (
    MATCHING_INPUT_DIM,
    NON_MATCHING_TYPE,
    NON_MATCHING_INPUT_SHAPE
)

NON_MATCHING_INPUT_TYPE = NON_MATCHING_TYPE
NON_MATCHING_OUTPUT_TYPE = [
    TensorType.INT8, TensorType.INT16, TensorType.UINT8, TensorType.FLOAT32
]
NON_MATCHING_OUTPUT_TYPE = [
    TensorType.INT8, TensorType.INT16, TensorType.UINT8, TensorType.FLOAT32
]
NON_MATCHING_AXIS_VALUE = [0, 2, 3]


def build_model(*, input_shape, input_type):
    model = XCOREModel()
    subgraph = model.create_subgraph()
    tin = subgraph.create_tensor(
        'input', type_=input_type, shape=input_shape, isinput=True)
    tout = subgraph.create_tensor(
        'output', TensorType.INT32, tin.shape, isoutput=True)
    op = subgraph.create_operator(OperatorCode(BuiltinOpCodes.ARG_MAX),
                                  inputs=[tin], outputs=[tout])

    dim_tensor = subgraph.create_tensor(
        f"{op.name}/axis", TensorType.INT32, shape=[])
    op.inputs.append(dim_tensor)
    dim_tensor.buffer.data = np.int32([1])

    return model


@pytest.fixture()
def trf_pass():
    return ReplaceArgMax16Pass()


@pytest.fixture(params=MATCHING_INPUT_DIM)
def input_dim(request):
    return request.param


@pytest.fixture()
def model(input_dim):
    input_shape = [1, input_dim]
    return build_model(input_shape=input_shape, input_type=TensorType.INT16)


def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[0])


@pytest.mark.parametrize('input_type', NON_MATCHING_INPUT_TYPE)
def test_non_matching_input_type(trf_pass, model, input_type):
    op = model.subgraphs[0].operators[0]
    op.inputs[0].type = input_type
    assert not trf_pass.match(op)


@pytest.mark.parametrize('output_type', NON_MATCHING_OUTPUT_TYPE)
def test_non_matching_output_type(trf_pass, model, output_type):
    op = model.subgraphs[0].operators[0]
    op.outputs[0].type = output_type
    assert not trf_pass.match(op)


@pytest.mark.parametrize('axis', NON_MATCHING_AXIS_VALUE)
def test_non_matching_axis_value(trf_pass, model, axis):
    op = model.subgraphs[0].operators[0]
    dim_tensor = op.inputs[1]
    dim_tensor.buffer.data = np.int32([axis])
    assert not trf_pass.match(op)


@pytest.mark.parametrize('input_shape', NON_MATCHING_INPUT_SHAPE)
def test_non_matching_input_shape(trf_pass, input_shape):
    model = build_model(input_shape=input_shape, input_type=TensorType.INT16)
    assert not trf_pass.match(model.subgraphs[0].operators[0])


def test_mutate(model, trf_pass):
    trf_pass.run(model)
    subgraph = model.subgraphs[0]
    assert subgraph.operators[-1].operator_code.code == XCOREOpCodes.XC_argmax_16

    # check input/output/intermediate tensors
    tin = subgraph.get_tensor('input')
    tout = subgraph.get_tensor('output')

    assert len(subgraph.operators) == 1
    assert len(subgraph.tensors) == 2
    assert tin in subgraph.inputs and tin not in subgraph.outputs
    assert tout in subgraph.outputs and tout not in subgraph.inputs


if __name__ == "__main__":
    pytest.main()
