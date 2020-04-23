# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType, OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import LegalizeFloatOutputPass

from .test_LegalizeFloatInputPass import simple_model, non_matching_model


@pytest.fixture()
def dual_output_model():
    model = XCOREModel()
    subgraph = model.create_subgraph()

    # TODO: add operator options to specify split axis and number
    qin = subgraph.create_tensor(
        "quantized_input", TensorType.INT8, [1, 5, 5, 4], isinput=True
    )
    qout1 = subgraph.create_tensor(
        "quantized_output_1", TensorType.INT8, [1, 5, 5, 2], isoutput=True
    )
    qout2 = subgraph.create_tensor(
        "quantized_output_2", TensorType.INT8, [1, 5, 5, 2], isoutput=True
    )
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.SPLIT), inputs=[qin], outputs=[qout1, qout2]
    )

    return model


@pytest.fixture()
def trf_pass():
    return LegalizeFloatOutputPass()


def test_match(simple_model, trf_pass):
    assert trf_pass.match(simple_model.subgraphs[0].outputs[0])


def test_mutate(simple_model, trf_pass):
    subgraph = simple_model.subgraphs[0]
    trf_pass.mutate(subgraph.outputs[0])
    subgraph.sanity_check()

    qin = subgraph.get_tensor("quantized_input")
    qout = subgraph.get_tensor("quantized_output")
    fout = subgraph.get_tensor("quantized_output_float")

    assert len(subgraph.operators) == 2
    assert len(subgraph.tensors) == 3
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1
    assert qin in subgraph.inputs and qin not in subgraph.outputs
    assert qout not in (subgraph.inputs + subgraph.outputs)
    assert fout in subgraph.outputs and fout not in subgraph.inputs


def test_run(simple_model, trf_pass):
    trf_pass.run(simple_model)
    simple_model.sanity_check()
    subgraph = simple_model.subgraphs[0]

    qin = subgraph.get_tensor("quantized_input")
    qout = subgraph.get_tensor("quantized_output")
    fout = subgraph.get_tensor("quantized_output_float")

    assert len(subgraph.operators) == 2
    assert len(subgraph.tensors) == 3
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1
    assert qin in subgraph.inputs and qin not in subgraph.outputs
    assert qout not in (subgraph.inputs + subgraph.outputs)
    assert fout in subgraph.outputs and fout not in subgraph.inputs


def test_run_dual_output(dual_output_model, trf_pass):
    trf_pass.run(dual_output_model)
    dual_output_model.sanity_check()
    subgraph = dual_output_model.subgraphs[0]

    qin = subgraph.get_tensor("quantized_input")
    qout_1 = subgraph.get_tensor("quantized_output_1")
    qout_2 = subgraph.get_tensor("quantized_output_2")
    fout_1 = subgraph.get_tensor("quantized_output_1_float")
    fout_2 = subgraph.get_tensor("quantized_output_2_float")

    assert len(subgraph.operators) == 3
    assert len(subgraph.tensors) == 5
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 2
    assert qin in subgraph.inputs and qin not in subgraph.outputs
    assert qout_1 not in (subgraph.inputs + subgraph.outputs)
    assert qout_2 not in (subgraph.inputs + subgraph.outputs)
    assert fout_1 in subgraph.outputs and fout_1 not in subgraph.inputs
    assert fout_2 in subgraph.outputs and fout_2 not in subgraph.inputs


def test_non_match(trf_pass, non_matching_model):
    for output_tensor in non_matching_model.subgraphs[0].outputs:
        assert not trf_pass.match(output_tensor)


if __name__ == "__main__":
    pytest.main()
