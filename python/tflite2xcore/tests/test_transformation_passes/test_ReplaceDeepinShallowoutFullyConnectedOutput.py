# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
from pytest_cases import pytest_fixture_plus, pytest_parametrize_plus, fixture_ref

import numpy

from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import ReplaceDeepinShallowoutFullyConnectedOutput


@pytest_fixture_plus(params=[(1, 4, 4, 8), (1, 32, 1, 1)])
def perceptron(request):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    input_shape = list(request.param)
    weight_shape = [10, numpy.prod(input_shape[1:])]
    tin = subgraph.create_tensor('input', TensorType.INT8, shape=input_shape, isinput=True)
    w = subgraph.create_tensor('weights', TensorType.INT8, shape=weight_shape,
                               quantization={'scale': [0.35], 'zero_point': [0]})
    b = subgraph.create_tensor('biases', TensorType.INT32, shape=weight_shape[:1])
    tout = subgraph.create_tensor('output', tin.type, shape=[1, weight_shape[0]], isoutput=True)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.FULLY_CONNECTED),
                             inputs=[tin, w, b], outputs=[tout])

    return model


@pytest_fixture_plus(params=[(64, 32, 10), (64, 64, 15), (64, 32, 2), (64, 64, 1)])
def mlp(request):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    input_nodes, hidden_nodes, output_nodes = request.param

    tin = subgraph.create_tensor('input', TensorType.INT8, shape=[1, input_nodes], isinput=True)
    w1 = subgraph.create_tensor('weights_1', TensorType.INT8, shape=[hidden_nodes, input_nodes],
                                quantization={'scale': [0.35], 'zero_point': [0]})
    b1 = subgraph.create_tensor('biases_1', TensorType.INT32, shape=[hidden_nodes])
    tmid = subgraph.create_tensor('intermediate', TensorType.INT8, shape=[1, hidden_nodes])
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.FULLY_CONNECTED),
                             inputs=[tin, w1, b1], outputs=[tmid])

    w2 = subgraph.create_tensor('weights_2', TensorType.INT8, shape=[output_nodes, hidden_nodes],
                                quantization={'scale': [0.22], 'zero_point': [0]})
    b2 = subgraph.create_tensor('biases_2', TensorType.INT32, shape=[output_nodes])
    tout = subgraph.create_tensor('output', TensorType.INT8, shape=[1, output_nodes])
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.FULLY_CONNECTED),
                             inputs=[tmid, w2, b2], outputs=[tout])

    return model


@pytest.fixture()
def trf_pass():
    return ReplaceDeepinShallowoutFullyConnectedOutput()


def test_nonmatch_mlp(mlp, trf_pass):
    assert not trf_pass.match(mlp.subgraphs[0].operators[0])


@pytest_parametrize_plus('model', [fixture_ref(perceptron), fixture_ref(mlp)])
def test_match(model, trf_pass):
    trf_pass.match(model.subgraphs[0].operators[-1])



if __name__ == "__main__":
    pytest.main()
