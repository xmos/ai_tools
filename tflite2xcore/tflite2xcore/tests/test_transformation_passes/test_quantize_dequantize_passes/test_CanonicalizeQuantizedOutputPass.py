# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.xcore_schema import TensorType, OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import CanonicalizeQuantizedOutputPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_split,
    build_dequantize,
    build_abs,
    _glue_ops,
)

from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_tensors,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

for params in PARAMS.values():
    params["non_matching_tensors"] = [
        {"input": tensor_type_dict["input"], "output": tensor_type_dict["input"]}
        for tensor_type_dict in params["non_matching_tensors"]
        if "input" in tensor_type_dict and len(tensor_type_dict) == 1
    ]

_NON_MATCHING_TENSORS = [
    {"output_dequantized": TensorType.INT8},
    {"output_dequantized": TensorType.INT16},
    {"output_dequantized": TensorType.INT32},
    {"output_dequantized": TensorType.UINT8},
]

PARAMS["default"].update({"num_splits": [2, 4]})
PARAMS["default"]["non_matching_tensors"].extend(_NON_MATCHING_TENSORS)

PARAMS["smoke"].update({"num_splits": [2]})
PARAMS["smoke"]["non_matching_tensors"].extend(_NON_MATCHING_TENSORS[::2])


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def model(input_shape):
    model = build_abs(input_shape=input_shape, tensor_type=TensorType.INT8)
    subgraph = model.subgraphs[0]

    build_dequantize(subgraph, input_shape=input_shape)

    _glue_ops(*subgraph.operators[:2])

    return model


@pytest.fixture()
def model_multi_out(input_shape, num_splits):
    model = build_split(
        input_shape=input_shape,
        num_splits=num_splits,
        tensor_type=TensorType.INT8,
        axis=2,
    )
    subgraph = model.subgraphs[0]

    op = model.subgraphs[0].operators[0]
    for qout in op.outputs:
        subgraph.outputs.remove(qout)
        fout = subgraph.create_tensor(
            qout.name + "_float", TensorType.FLOAT32, qout.shape, isoutput=True
        )
        subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qout], outputs=[fout]
        )

    return model


@pytest.fixture()
def model_non_matching_consumer(model, input_shape):
    subgraph = model.subgraphs[0]
    op_deq = subgraph.operators[-1]

    build_abs(subgraph, input_shape=input_shape, tensor_type=op_deq.outputs[0].type)
    op_abs_2 = subgraph.operators[-1]
    _glue_ops(op_deq, op_abs_2)  # this removes op_deq.outputs[0]
    subgraph.outputs.append(op_abs_2.inputs[0])  # so we put back its replacement

    return model


@pytest.fixture()
def trf_pass():
    return CanonicalizeQuantizedOutputPass()


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(model, trf_pass):
    subgraph = model.subgraphs[0]
    trf_pass.mutate(subgraph.operators[1])
    subgraph.sanity_check()

    qin = subgraph.get_tensor("input")
    qout = subgraph.get_tensor("output")

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.ABS
    assert len(subgraph.tensors) == 2
    assert qin in subgraph.inputs
    assert qin not in subgraph.outputs
    assert qout in subgraph.outputs
    assert qout not in subgraph.inputs


def test_multi_out(model_multi_out, num_splits, trf_pass):
    trf_pass.run(model_multi_out)
    model_multi_out.sanity_check()
    subgraph = model_multi_out.subgraphs[0]

    assert len(subgraph.operators) == 1
    assert subgraph.operators[0].operator_code.code is BuiltinOpCodes.SPLIT
    assert len(subgraph.tensors) == 2 + num_splits  # split has two inputs

    taxis = subgraph.get_tensor("axis")
    assert taxis not in subgraph.outputs
    assert taxis not in subgraph.inputs

    tin = subgraph.get_tensor("input")
    assert tin in subgraph.inputs
    assert tin not in subgraph.outputs

    assert len(subgraph.outputs) == num_splits
    for j, tout in enumerate(subgraph.outputs):
        assert tout not in subgraph.inputs, f"subgraph.outputs[{j}]"


def test_non_matching_input(trf_pass, input_shape):
    # NOTE: a single DEQUANTIZE will always have an input tensor that is an input
    #       to the subgraph, hence it should not be matched
    model = build_dequantize(input_shape=input_shape)
    _test_non_matching_params(trf_pass, model)


def test_non_matching_consumers(trf_pass, model_non_matching_consumer):
    for j, op in enumerate(model_non_matching_consumer.subgraphs[0].operators):
        assert not trf_pass.match(op), f"subgraphs[0].operators[{j}]"


if __name__ == "__main__":
    pytest.main()
