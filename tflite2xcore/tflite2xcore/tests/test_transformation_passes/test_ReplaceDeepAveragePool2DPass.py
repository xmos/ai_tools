# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from pytest_cases import pytest_fixture_plus, pytest_parametrize_plus, fixture_ref
from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import ReplaceDeepAveragePool2DPass

# TODO: fix this when refactoring model builder functions
"""from .test_ReplaceDeepinAnyoutFullyConnectedOutputPass import (
    matching_perceptron,
    non_matching_shape_perceptron,
    mlp,
    non_matching_mlp
)"""


def build_avgpool_model(input_shape):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    output_shape = [input_shape[0], input_shape[1] / 2, input_shape[1] / 2, input_shape[3]]
    tin = subgraph.create_tensor('input', TensorType.INT8, shape=input_shape, isinput=True)
    tout = subgraph.create_tensor('output', tin.type, shape=output_shape, isoutput=True)

    op = subgraph.create_operator(OperatorCode(BuiltinOpCodes.AVERAGE_POOL_2D),
                                  inputs=[tin], outputs=[tout])
    op.builtin_options = {'padding': 'VALID',
                          'stride_w': 2, 'stride_h': 2,
                          'filter_height': 2, 'filter_width': 2,
                          'fused_activation_function': 'NONE'}

    return model


@pytest_fixture_plus(params=[
    (1, 2, 2, 32), (1, 4, 4, 64), (1, 6, 24, 32), (1, 16, 8, 64)
])
def matching(request):
    return build_avgpool_model(input_shape=list(request.param))


@pytest_fixture_plus(params=[
    (1, 2, 2, 16), (1, 4, 4, 35), (1, 5, 24, 32), (1, 16, 9, 64)
])
def non_matching_shape(request):
    return build_avgpool_model(input_shape=list(request.param))


@pytest_fixture_plus(params=[
    ('padding', 'SAME'),
    ('fused_activation_function', 'RELU'),
    ('fused_activation_function', 'RELU6'),
    ('stride_w', 1),
    ('stride_h', 1),
    ('filter_width', 3),
    ('filter_height', 3)
])
def non_matching_options(matching, request):
    subgraph = matching.subgraphs[0]
    op = subgraph.operators[-1]
    op.builtin_options[request.param[0]] = request.param[1]
    return matching


@pytest.fixture()
def trf_pass():
    return ReplaceDeepAveragePool2DPass()


@pytest_parametrize_plus('model', [
    fixture_ref(matching),
])
def test_match(model, trf_pass):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


@pytest_parametrize_plus('model', [
    fixture_ref(non_matching_options),
    fixture_ref(non_matching_shape),
])
def test_non_match_conv(model, trf_pass):
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


# TODO: fix this when refactoring model builder functions
"""@pytest_parametrize_plus('model', [
    fixture_ref(matching_perceptron),
    fixture_ref(non_matching_shape_perceptron),
    fixture_ref(mlp),
    fixture_ref(non_matching_mlp)
])
def test_non_match_ops(model, trf_pass):
    for op in model.subgraphs[0].operators:
        assert not trf_pass.match(op)"""


if __name__ == "__main__":
    pytest.main()
