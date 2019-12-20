# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from pytest_cases import pytest_fixture_plus, pytest_parametrize_plus, fixture_ref
from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import ReplaceDeepinDeepoutConv2DPass


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


@pytest.fixture()
def trf_pass():
    return ReplaceDeepinDeepoutConv2DPass()


@pytest_parametrize_plus('model', [fixture_ref(matching)])
def test_match(model, trf_pass):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


if __name__ == "__main__":
    pytest.main()
