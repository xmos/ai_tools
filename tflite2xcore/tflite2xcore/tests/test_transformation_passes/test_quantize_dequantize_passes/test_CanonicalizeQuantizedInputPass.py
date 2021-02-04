# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from tflite2xcore.xcore_schema import (
    XCOREModel,
    Subgraph,
    TensorType,
    OperatorCode,
    BuiltinOpCodes,
)
from tflite2xcore.transformation_passes import CanonicalizeQuantizedInputPass


@pytest.fixture()
def simple_model():
    subgraph = Subgraph(model=XCOREModel())

    fin = subgraph.create_tensor(
        "input", TensorType.FLOAT32, [1, 5, 5, 3], isinput=True
    )
    qin = subgraph.create_tensor("quantized_input", TensorType.INT8, fin.shape)
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin], outputs=[qin]
    )

    qout = subgraph.create_tensor(
        "quantized_output", TensorType.INT8, qin.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.ABS), inputs=[qin], outputs=[qout]
    )

    return subgraph.model


@pytest.fixture()
def dual_input_model():
    subgraph = Subgraph(model=XCOREModel())

    fin1 = subgraph.create_tensor(
        "input_1", TensorType.FLOAT32, [1, 5, 5, 3], isinput=True
    )
    qin1 = subgraph.create_tensor("quantized_input_1", TensorType.INT8, fin1.shape)
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin1], outputs=[qin1]
    )

    fin2 = subgraph.create_tensor(
        "input_2", TensorType.FLOAT32, fin1.shape, isinput=True
    )
    qin2 = subgraph.create_tensor("quantized_input_2", TensorType.INT8, fin2.shape)
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin2], outputs=[qin2]
    )

    qout = subgraph.create_tensor(
        "quantized_output", TensorType.INT8, qin1.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.ADD), inputs=[qin1, qin2], outputs=[qout]
    )

    return subgraph.model


@pytest.fixture()
def non_matching_model():
    subgraph = Subgraph(model=XCOREModel())

    fin1 = subgraph.create_tensor(
        "input_1", TensorType.FLOAT32, [1, 5, 5, 3], isinput=True
    )
    qout1 = subgraph.create_tensor(
        "quantized_output_1", TensorType.INT8, fin1.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin1], outputs=[qout1]
    )

    fin2 = subgraph.create_tensor(
        "input_2", TensorType.FLOAT32, [1, 3, 3, 8], isinput=True
    )
    qout2 = subgraph.create_tensor(
        "quantized_output_2", TensorType.INT8, fin2.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin2], outputs=[qout2]
    )

    return subgraph.model


@pytest.fixture()
def trf_pass():
    return CanonicalizeQuantizedInputPass()


def test_match(simple_model, trf_pass):
    assert trf_pass.match(simple_model.subgraphs[0].operators[0])


def test_mutate(simple_model, trf_pass):
    subgraph = simple_model.subgraphs[0]
    trf_pass.mutate(subgraph.operators[0])
    subgraph.sanity_check()

    qin = subgraph.get_tensor("quantized_input")
    qout = subgraph.get_tensor("quantized_output")

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.ABS
    assert len(subgraph.tensors) == 2
    assert qin in subgraph.inputs and qin not in subgraph.outputs
    assert qout in subgraph.outputs and qout not in subgraph.inputs


def test_run_simple(simple_model, trf_pass):
    trf_pass.run(simple_model)
    simple_model.sanity_check()
    subgraph = simple_model.subgraphs[0]

    qin = subgraph.get_tensor("quantized_input")
    qout = subgraph.get_tensor("quantized_output")

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.ABS
    assert len(subgraph.tensors) == 2
    assert qin in subgraph.inputs and qin not in subgraph.outputs
    assert qout in subgraph.outputs and qout not in subgraph.inputs


def test_run_dual_input(dual_input_model, trf_pass):
    trf_pass.run(dual_input_model)
    dual_input_model.sanity_check()
    subgraph = dual_input_model.subgraphs[0]

    qin1 = subgraph.get_tensor("quantized_input_1")
    qin2 = subgraph.get_tensor("quantized_input_2")
    qout = subgraph.get_tensor("quantized_output")

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.ADD
    assert len(subgraph.tensors) == 3
    assert qin1 in subgraph.inputs and qin1 not in subgraph.outputs
    assert qin2 in subgraph.inputs and qin2 not in subgraph.outputs
    assert qout in subgraph.outputs and qout not in subgraph.inputs


def test_non_match(trf_pass, non_matching_model):
    for op in non_matching_model.subgraphs[0].operators:
        assert not trf_pass.match(op)


if __name__ == "__main__":
    pytest.main()
