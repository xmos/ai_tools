# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy
import pytest

from tflite2xcore.xcore_model import TensorType
from tflite2xcore.transformation_passes import ReplaceFullyConnectedIntermediatePass

from .model_builders import (
    build_fc, build_mlp, build_softmax
)


MATCHING_INPUT_SIZE = [
    (1, 1, 32), (2, 2, 8), (8, 8, 1), (64,),
    (1, 1, 31), (2, 2, 7), (9, 4, 6), (63,)
]
MATCHING_OUTPUTS = [1, 2, 10, 15, 16, 17, 100]
MATCHING_HIDDEN_NODES = MATCHING_OUTPUTS + [numpy.prod(t) for t in MATCHING_INPUT_SIZE]

NON_MATCHING_TENSORS = ('tensor_name', 'new_type'), [
    ('input', TensorType.INT16), ('input', TensorType.INT32),
    ('weights_1', TensorType.INT16), ('weights_1', TensorType.INT32),
    ('biases_1', TensorType.INT8), ('biases_1', TensorType.INT16),
    ('intermediate', TensorType.INT16), ('intermediate', TensorType.INT32)
]


@pytest.fixture()
def trf_pass():
    return ReplaceFullyConnectedIntermediatePass()


@pytest.fixture(params=MATCHING_OUTPUTS)
def outputs(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_SIZE)
def input_size(request):
    return request.param


@pytest.fixture(params=MATCHING_HIDDEN_NODES)
def hidden_nodes(request):
    return request.param


@pytest.fixture()
def mlp(outputs, hidden_nodes, input_size):
    return build_mlp(outputs=outputs, hidden_nodes=hidden_nodes, input_size=input_size)


@pytest.fixture()
def logistic(outputs, input_size):
    return build_softmax(outputs=outputs, input_size=input_size)


@pytest.fixture()
def fc_model(outputs, input_size):
    return build_fc(outputs=outputs, input_size=input_size)


def test_mlp_input_match(trf_pass, mlp):
    assert trf_pass.match(mlp.subgraphs[0].operators[0])


def test_mlp_output_non_match(trf_pass, mlp):
    assert not trf_pass.match(mlp.subgraphs[0].operators[-1])


def test_logistic(logistic, trf_pass):
    assert trf_pass.match(logistic.subgraphs[0].operators[0])
    assert not trf_pass.match(logistic.subgraphs[0].operators[-1])


def test_fc_non_match(fc_model, trf_pass):
    assert not trf_pass.match(fc_model.subgraphs[0].operators[-1])


@pytest.mark.parametrize(*NON_MATCHING_TENSORS)
def test_non_matching_logistic_input_types(trf_pass, logistic, tensor_name, new_type):
    subgraph = logistic.subgraphs[0]
    subgraph.get_tensor(tensor_name).type = new_type
    assert not trf_pass.match(logistic.subgraphs[0].operators[0])
    assert not trf_pass.match(logistic.subgraphs[0].operators[-1])


@pytest.mark.parametrize(*NON_MATCHING_TENSORS)
def test_non_matching_mlp_input_types(trf_pass, mlp, tensor_name, new_type):
    subgraph = mlp.subgraphs[0]
    subgraph.get_tensor(tensor_name).type = new_type
    assert not trf_pass.match(mlp.subgraphs[0].operators[0])
    assert not trf_pass.match(mlp.subgraphs[0].operators[-1])


if __name__ == "__main__":
    pytest.main()
