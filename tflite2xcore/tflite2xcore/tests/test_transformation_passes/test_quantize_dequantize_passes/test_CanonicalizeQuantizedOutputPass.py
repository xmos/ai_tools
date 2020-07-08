# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType, OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import CanonicalizeQuantizedOutputPass


@pytest.fixture()
def simple_model():
    model = XCOREModel()
    subgraph = model.create_subgraph()

    qin = subgraph.create_tensor(
        "quantized_input", TensorType.INT8, [1, 5, 5, 3], isinput=True
    )
    qout = subgraph.create_tensor("quantized_output", TensorType.INT8, qin.shape)
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.ABS), inputs=[qin], outputs=[qout]
    )

    fout = subgraph.create_tensor(
        "output", TensorType.FLOAT32, qout.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qout], outputs=[fout]
    )

    return model


@pytest.fixture()
def dual_output_model():
    model = XCOREModel()
    subgraph = model.create_subgraph()

    # TODO: add operator options to specify split axis and number
    qin = subgraph.create_tensor(
        "quantized_input", TensorType.INT8, [1, 5, 5, 4], isinput=True
    )
    qout1 = subgraph.create_tensor("quantized_output_1", TensorType.INT8, [1, 5, 5, 2])
    qout2 = subgraph.create_tensor("quantized_output_2", TensorType.INT8, [1, 5, 5, 2])
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.SPLIT), inputs=[qin], outputs=[qout1, qout2]
    )

    fout1 = subgraph.create_tensor(
        "output_1", TensorType.FLOAT32, qout1.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qout1], outputs=[fout1]
    )

    fout2 = subgraph.create_tensor(
        "output_2", TensorType.FLOAT32, qout1.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qout2], outputs=[fout2]
    )

    return model


@pytest.fixture()
def non_matching_input_model():
    model = XCOREModel()
    subgraph = model.create_subgraph()

    qin1 = subgraph.create_tensor(
        "quantized_input", TensorType.INT8, [1, 5, 5, 3], isinput=True
    )
    fout1 = subgraph.create_tensor(
        "output", TensorType.FLOAT32, qin1.shape, isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qin1], outputs=[fout1]
    )

    return model


@pytest.fixture()
def trf_pass():
    return CanonicalizeQuantizedOutputPass()


def test_match(simple_model, trf_pass):
    assert trf_pass.match(simple_model.subgraphs[0].operators[1])


def test_mutate(simple_model, trf_pass):
    subgraph = simple_model.subgraphs[0]
    trf_pass.mutate(subgraph.operators[1])
    subgraph.sanity_check()

    qin = subgraph.get_tensor("quantized_input")
    qout = subgraph.get_tensor("quantized_output")

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.ABS
    assert len(subgraph.tensors) == 2
    assert qin in subgraph.inputs
    assert qin not in subgraph.outputs
    assert qout in subgraph.outputs
    assert qout not in subgraph.inputs


def test_run_simple(simple_model, trf_pass):
    trf_pass.run(simple_model)
    simple_model.sanity_check()
    subgraph = simple_model.subgraphs[0]

    qin = subgraph.get_tensor("quantized_input")
    qout = subgraph.get_tensor("quantized_output")

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.ABS
    assert len(subgraph.tensors) == 2
    assert qin in subgraph.inputs
    assert qin not in subgraph.outputs
    assert qout in subgraph.outputs
    assert qout not in subgraph.inputs


def test_run_dual_output(dual_output_model, trf_pass):
    trf_pass.run(dual_output_model)
    dual_output_model.sanity_check()
    subgraph = dual_output_model.subgraphs[0]

    qin = subgraph.get_tensor("quantized_input")
    qout_1 = subgraph.get_tensor("quantized_output_1")
    qout_2 = subgraph.get_tensor("quantized_output_2")

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.SPLIT
    assert len(subgraph.tensors) == 3
    assert qin in subgraph.inputs
    assert qin not in subgraph.outputs
    assert qout_1 in subgraph.outputs
    assert qout_1 not in subgraph.inputs
    assert qout_2 in subgraph.outputs
    assert qout_2 not in subgraph.inputs


def test_non_matching_input(trf_pass, non_matching_input_model):
    for op in non_matching_input_model.subgraphs[0].operators:
        assert not trf_pass.match(op)


if __name__ == "__main__":
    pytest.main()
