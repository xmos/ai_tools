# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType, OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import RemoveSoftmaxOutputPass


@pytest.fixture(params=[TensorType.FLOAT32, TensorType.INT8])
def simple_model(request):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    tin = subgraph.create_tensor(
        "input", type_=request.param, shape=[1, 5, 5, 3], isinput=True
    )
    pre_out = subgraph.create_tensor("pre_output", tin.type, tin.shape)
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.ABS), inputs=[tin], outputs=[pre_out]
    )

    tout = subgraph.create_tensor("output", pre_out.type, pre_out.shape, isoutput=True)
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.SOFTMAX), inputs=[pre_out], outputs=[tout]
    )

    return model


@pytest.fixture(params=[TensorType.FLOAT32, TensorType.INT8])
def dual_output_model():
    model = XCOREModel()
    subgraph = model.create_subgraph()

    # TODO: add operator options to specify split axis and number
    tin = subgraph.create_tensor("input", TensorType.INT8, [1, 5, 5, 4], isinput=True)
    pre_out1 = subgraph.create_tensor("pre_output_1", tin.type, [1, 5, 5, 2])
    pre_out2 = subgraph.create_tensor("pre_output_2", tin.type, [1, 5, 5, 2])
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.SPLIT), inputs=[tin], outputs=[pre_out1, pre_out2]
    )

    out1 = subgraph.create_tensor(
        "output_1", pre_out1.type, pre_out1.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.SOFTMAX), inputs=[pre_out1], outputs=[out1]
    )

    out2 = subgraph.create_tensor(
        "output_2", pre_out2.type, pre_out2.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.SOFTMAX), inputs=[pre_out2], outputs=[out2]
    )

    return model


@pytest.fixture()
def non_matching_model():
    model = XCOREModel()
    subgraph = model.create_subgraph()

    fin1 = subgraph.create_tensor(
        "input_1", TensorType.FLOAT32, [1, 5, 5, 3], isinput=True
    )
    qout1 = subgraph.create_tensor(
        "quantized_output_1", TensorType.INT8, fin1.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin1], outputs=[qout1]
    )

    qin2 = subgraph.create_tensor(
        "quantized_input_2", TensorType.INT8, [1, 3, 3, 8], isinput=True
    )
    fout2 = subgraph.create_tensor(
        "output_2", TensorType.FLOAT32, qin2.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qin2], outputs=[fout2]
    )

    return model


@pytest.fixture()
def trf_pass():
    return RemoveSoftmaxOutputPass()


def test_match(simple_model, trf_pass):
    assert trf_pass.match(simple_model.subgraphs[0].operators[1])


def test_mutate(simple_model, trf_pass):
    subgraph = simple_model.subgraphs[0]
    trf_pass.mutate(subgraph.operators[1])
    subgraph.sanity_check()

    tin = subgraph.get_tensor("input")
    tout = subgraph.get_tensor("pre_output")

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.ABS
    assert len(subgraph.tensors) == 2
    assert tin in subgraph.inputs and tin not in subgraph.outputs
    assert tout in subgraph.outputs and tout not in subgraph.inputs


def test_run_simple(simple_model, trf_pass):
    trf_pass.run(simple_model)
    simple_model.sanity_check()
    subgraph = simple_model.subgraphs[0]

    tin = subgraph.get_tensor("input")
    tout = subgraph.get_tensor("pre_output")

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.ABS
    assert len(subgraph.tensors) == 2
    assert tin in subgraph.inputs and tin not in subgraph.outputs
    assert tout in subgraph.outputs and tout not in subgraph.inputs


def test_run_dual_output(dual_output_model, trf_pass):
    trf_pass.run(dual_output_model)
    dual_output_model.sanity_check()
    subgraph = dual_output_model.subgraphs[0]

    tin = subgraph.get_tensor("input")
    out_1 = subgraph.get_tensor("pre_output_1")
    out_2 = subgraph.get_tensor("pre_output_2")

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.SPLIT
    assert len(subgraph.tensors) == 3
    assert tin in subgraph.inputs and tin not in subgraph.outputs
    assert out_1 in subgraph.outputs and out_1 not in subgraph.inputs
    assert out_2 in subgraph.outputs and out_2 not in subgraph.inputs


def test_non_match(trf_pass, non_matching_model):
    for op in non_matching_model.subgraphs[0].operators:
        assert not trf_pass.match(op)


if __name__ == "__main__":
    pytest.main()
