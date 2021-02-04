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
from tflite2xcore.transformation_passes import LegalizeFloatInputPass


@pytest.fixture()
def simple_model():
    subgraph = Subgraph(model=XCOREModel())

    qin = subgraph.create_tensor(
        "quantized_input", TensorType.INT8, [1, 5, 5, 3], isinput=True
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

    qin1 = subgraph.create_tensor(
        "quantized_input_1", TensorType.INT8, [1, 5, 5, 3], isinput=True
    )
    qin2 = subgraph.create_tensor(
        "quantized_input_2", TensorType.INT8, qin1.shape, isinput=True
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

    qin = subgraph.create_tensor(
        "quantized_input", TensorType.INT32, [1, 5, 5, 3], isinput=True
    )
    qout = subgraph.create_tensor(
        "quantized_output", TensorType.INT32, qin.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.ABS), inputs=[qin], outputs=[qout]
    )

    fin = subgraph.create_tensor(
        "float_input", TensorType.FLOAT32, [1, 3, 3, 8], isinput=True
    )
    fout = subgraph.create_tensor(
        "float_output", TensorType.FLOAT32, fin.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.ABS), inputs=[fin], outputs=[fout]
    )

    return subgraph.model


@pytest.fixture()
def trf_pass():
    return LegalizeFloatInputPass()


def test_match(simple_model, trf_pass):
    assert trf_pass.match(simple_model.subgraphs[0].inputs[0])


def test_mutate(simple_model, trf_pass):
    subgraph = simple_model.subgraphs[0]
    trf_pass.mutate(subgraph.inputs[0])
    simple_model.sanity_check()

    qin = subgraph.get_tensor("quantized_input")
    fin = subgraph.get_tensor("quantized_input_float")
    qout = subgraph.get_tensor("quantized_output")

    assert len(subgraph.operators) == 2
    assert len(subgraph.tensors) == 3
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1
    assert fin in subgraph.inputs and fin not in subgraph.outputs
    assert qin not in (subgraph.inputs + subgraph.outputs)
    assert qout in subgraph.outputs and qout not in subgraph.inputs


def test_run_simple(simple_model, trf_pass):
    trf_pass.run(simple_model)
    simple_model.sanity_check()
    subgraph = simple_model.subgraphs[0]

    qin = subgraph.get_tensor("quantized_input")
    fin = subgraph.get_tensor("quantized_input_float")
    qout = subgraph.get_tensor("quantized_output")

    assert len(subgraph.operators) == 2
    assert len(subgraph.tensors) == 3
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1
    assert fin in subgraph.inputs and fin not in subgraph.outputs
    assert qin not in (subgraph.inputs + subgraph.outputs)
    assert qout in subgraph.outputs and qout not in subgraph.inputs


def test_run_dual_input(dual_input_model, trf_pass):
    trf_pass.run(dual_input_model)
    dual_input_model.sanity_check()
    subgraph = dual_input_model.subgraphs[0]

    qin1 = subgraph.get_tensor("quantized_input_1")
    qin2 = subgraph.get_tensor("quantized_input_2")
    fin1 = subgraph.get_tensor("quantized_input_1_float")
    fin2 = subgraph.get_tensor("quantized_input_2_float")
    qout = subgraph.get_tensor("quantized_output")

    assert len(subgraph.operators) == 3
    assert len(subgraph.tensors) == 5
    assert len(subgraph.inputs) == 2
    assert len(subgraph.outputs) == 1
    assert qin1 not in (subgraph.inputs + subgraph.outputs)
    assert qin2 not in (subgraph.inputs + subgraph.outputs)
    assert fin1 in subgraph.inputs and fin1 not in subgraph.outputs
    assert fin1 in subgraph.inputs and fin2 not in subgraph.outputs
    assert qout in subgraph.outputs and qout not in subgraph.inputs


def test_non_match(trf_pass, non_matching_model):
    for input_tensor in non_matching_model.subgraphs[0].inputs:
        assert not trf_pass.match(input_tensor)


if __name__ == "__main__":
    pytest.main()
