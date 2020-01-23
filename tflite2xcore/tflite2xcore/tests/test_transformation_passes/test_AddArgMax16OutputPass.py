# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import AddArgMax16OutputPass

MATCHING_INPUT_DIM = [2, 5, 10, 100]

NON_MATCHING_TYPE = [TensorType.INT8, TensorType.FLOAT32, TensorType.INT32]
NON_MATCHING_OUTPUTS = [2, 3]
NON_MATCHING_INPUT_SHAPE = [(1, 10, 1), (1, 10, 1, 1), (1, 3, 3, 8), (1, 5, 5, 3)]


def build_model(*, input_shape, tensor_type):
    model = XCOREModel()
    subgraph = model.create_subgraph()
    tin = subgraph.create_tensor(
        'input', type_=tensor_type, shape=input_shape, isinput=True)
    tout = subgraph.create_tensor(
        'output', tin.type, tin.shape, isoutput=True)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.ABS),
                             inputs=[tin], outputs=[tout])

    return model


@pytest.fixture()
def trf_pass():
    return AddArgMax16OutputPass()


@pytest.fixture(params=MATCHING_INPUT_DIM)
def input_dim(request):
    return request.param


@pytest.fixture()
def model(input_dim):
    input_shape = [1, input_dim]
    return build_model(input_shape=input_shape, tensor_type=TensorType.INT16)


def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].outputs[0])


@pytest.mark.parametrize('tensor_type', NON_MATCHING_TYPE)
def test_non_matching_type(trf_pass, model, tensor_type):
    op = model.subgraphs[0].operators[0]
    op.outputs[0].type = tensor_type
    assert not trf_pass.match(model.subgraphs[0].outputs[0])


@pytest.mark.parametrize('input_shape', NON_MATCHING_INPUT_SHAPE)
def test_non_matching_shape(trf_pass, input_shape):
    model = build_model(input_shape=input_shape, tensor_type=TensorType.INT16)
    assert not trf_pass.match(model.subgraphs[0].outputs[0])


@pytest.mark.parametrize('outputs', NON_MATCHING_OUTPUTS)
def test_non_matching_outputs(trf_pass, model, outputs):
    subgraph = model.subgraphs[0]
    op = subgraph.operators[0]
    output_tensor = op.outputs[0]

    for j in range(1, outputs):
        subgraph.create_tensor(
            f'output_{j}', output_tensor.type, output_tensor.shape, isoutput=True)

    assert not trf_pass.match(model.subgraphs[0].outputs[0])


def test_mutate(model, trf_pass):
    trf_pass.run(model)
    subgraph = model.subgraphs[0]
    assert subgraph.operators[-1].operator_code.code == BuiltinOpCodes.ARG_MAX

    # check input/output/intermediate tensors
    tin = subgraph.get_tensor('input')
    pre_out = subgraph.get_tensor('output')
    tout = subgraph.get_tensor('output_argmax')

    assert len(subgraph.operators) == 2
    assert len(subgraph.tensors) == 4
    assert tin in subgraph.inputs and tin not in subgraph.outputs
    assert pre_out not in (subgraph.inputs + subgraph.outputs)
    assert tout in subgraph.outputs and tout not in subgraph.inputs

    # check axis dim tensor
    argmax_input_tensors = subgraph.operators[-1].inputs
    assert len(argmax_input_tensors) == 2

    dim_tensor = argmax_input_tensors[1]
    assert dim_tensor.type == TensorType.INT32
    assert dim_tensor.shape == []
    assert dim_tensor.numpy == 1


if __name__ == "__main__":
    pytest.main()
