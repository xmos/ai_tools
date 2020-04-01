# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy
import pytest

from tflite2xcore.xcore_model import TensorType
from tflite2xcore.transformation_passes import ReplaceFullyConnectedOutputPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_mlp

from .test_ReplaceFullyConnectedIntermediatePass import (
    MATCHING_INPUT_SIZE,
    outputs, input_shape,
    fc_model, logistic
)
MATCHING_HIDDEN_NODES = [numpy.prod(t) for t in MATCHING_INPUT_SIZE]

NON_MATCHING_TENSORS = ('tensor_name', 'new_type'), [
    ('input', TensorType.INT16), ('input', TensorType.INT32),
    ('weights', TensorType.INT16), ('weights', TensorType.INT32),
    ('biases', TensorType.INT8), ('biases', TensorType.INT16),
    ('output', TensorType.INT16), ('output', TensorType.INT32)
]


@pytest.fixture()
def trf_pass():
    return ReplaceFullyConnectedOutputPass()


@pytest.fixture(params=MATCHING_HIDDEN_NODES)
def hidden_nodes(request):
    return request.param


@pytest.fixture()
def mlp(outputs, hidden_nodes, input_shape):
    return build_mlp(outputs=outputs, hidden_nodes=hidden_nodes, input_shape=input_shape)


def test_matching_params(trf_pass, fc_model):
    assert trf_pass.match(fc_model.subgraphs[0].operators[-1])


@pytest.mark.parametrize(*NON_MATCHING_TENSORS)
def test_non_matching_types(trf_pass, fc_model, tensor_name, new_type):
    subgraph = fc_model.subgraphs[0]
    subgraph.get_tensor(tensor_name).type = new_type
    assert not trf_pass.match(fc_model.subgraphs[0].operators[-1])


def test_logistic(logistic, trf_pass):
    assert not trf_pass.match(logistic.subgraphs[0].operators[0])
    assert not trf_pass.match(logistic.subgraphs[0].operators[-1])


def test_mlp_input_non_match(trf_pass, mlp):
    assert not trf_pass.match(mlp.subgraphs[0].operators[0])


def test_mlp_output_match(trf_pass, mlp):
    assert trf_pass.match(mlp.subgraphs[0].operators[-1])


if __name__ == "__main__":
    pytest.main()
