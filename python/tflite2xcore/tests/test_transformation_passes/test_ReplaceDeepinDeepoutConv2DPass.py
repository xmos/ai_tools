# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from pytest_cases import pytest_fixture_plus, pytest_parametrize_plus, fixture_ref
from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import ReplaceDeepinDeepoutConv2DPass
from .test_ReplaceDeepinShallowoutFullyConnectedOutputPass import (
    matching_perceptron,
    non_matching_perceptron,
    mlp,
    non_matching_mlp
)


@pytest_fixture_plus(params=[
    (16, 1, 1, 32), (16, 3, 3, 32), (32, 5, 5, 32), (16, 3, 7, 32)
])
def matching(request):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    weight_shape = list(request.param)
    input_shape = [1, 10, 10, weight_shape[-1]]
    tin = subgraph.create_tensor('input', TensorType.INT8, shape=input_shape, isinput=True)
    w = subgraph.create_tensor('weights', TensorType.INT8, shape=weight_shape)
    b = subgraph.create_tensor('biases', TensorType.INT32, shape=weight_shape[:1])
    tout = subgraph.create_tensor('output', tin.type, shape=[1, weight_shape[0]], isoutput=True)

    op = subgraph.create_operator(OperatorCode(BuiltinOpCodes.CONV_2D),
                                  inputs=[tin, w, b], outputs=[tout])
    op.builtin_options = {'padding': 'SAME',
                          'stride_w': 1, 'stride_h': 1,
                          'dilation_w_factor': 1, 'dilation_h_factor': 1}

    return model


@pytest_fixture_plus(params=[
    ('padding', 'VALID'),
    ('stride_w', 2),
    ('stride_h', 4),
    ('dilation_w_factor', 2),
    ('dilation_h_factor', 3)
])
def non_matching_conv(matching, request):
    op = matching.subgraphs[0].operators[0]
    op.builtin_options[request.param[0]] = request.param[1]
    return matching


@pytest.fixture()
def trf_pass():
    return ReplaceDeepinDeepoutConv2DPass()


@pytest_parametrize_plus('model', [fixture_ref(matching)])
def test_match(model, trf_pass):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


def test_non_match_conv(non_matching_conv, trf_pass):
    assert not trf_pass.match(non_matching_conv.subgraphs[0].operators[-1])


@pytest_parametrize_plus('model', [
    fixture_ref(matching_perceptron),
    fixture_ref(non_matching_perceptron),
    fixture_ref(mlp),
    fixture_ref(non_matching_mlp)
])
def test_non_match_ops(model, trf_pass):
    for op in model.subgraphs[0].operators:
        assert not trf_pass.match(op)


if __name__ == "__main__":
    pytest.main()
