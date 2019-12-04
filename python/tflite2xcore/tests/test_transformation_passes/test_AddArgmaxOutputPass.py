# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from xcore_model import XCOREModel
from operator_codes import OperatorCode, BuiltinOpCodes, XCOREOpCodes
from transformation_passes import AddArgmaxOutputPass


@pytest.fixture()
def simple_model():
    model = XCOREModel()
    subgraph = model.create_subgraph()

    tin = subgraph.create_tensor('input', type_='INT16', shape=[1, 10], isinput=True)
    tout = subgraph.create_tensor('output', tin.type, tin.shape, isoutput=True)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.ABS),
                             inputs=[tin], outputs=[tout])

    return model


@pytest.fixture()
def dual_output_model():
    model = XCOREModel()
    subgraph = model.create_subgraph()

    # TODO: add operator options to specify split axis and number
    qin = subgraph.create_tensor('quantized_input', 'INT16', [1, 20], isinput=True)
    qout1 = subgraph.create_tensor('quantized_output_1', 'INT16', [1, 10], isoutput=True)
    qout2 = subgraph.create_tensor('quantized_output_2', 'INT16', [1, 10], isoutput=True)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.SPLIT),
                             inputs=[qin], outputs=[qout1, qout2])

    return model


@pytest.fixture()
def trf_pass():
    return AddArgmaxOutputPass()


def test_match(simple_model, trf_pass):
    assert trf_pass.match(simple_model.subgraphs[0].outputs[0])


def test_mutate(simple_model, trf_pass):
    subgraph = simple_model.subgraphs[0]
    trf_pass.mutate(subgraph.outputs[0])

    tin = subgraph.get_tensor('input')
    pre_out = subgraph.get_tensor('output')
    tout = subgraph.get_tensor('output_argmax')

    assert len(subgraph.operators) == 2
    assert len(subgraph.tensors) == 3
    assert tin in subgraph.inputs and tin not in subgraph.outputs
    assert pre_out not in (subgraph.inputs + subgraph.outputs)
    assert tout in subgraph.outputs and tout not in subgraph.inputs


def test_run_simple(simple_model, trf_pass):
    trf_pass.run(simple_model)
    subgraph = simple_model.subgraphs[0]

    tin = subgraph.get_tensor('input')
    pre_out = subgraph.get_tensor('output')
    tout = subgraph.get_tensor('output_argmax')

    assert len(subgraph.operators) == 2
    assert len(subgraph.tensors) == 3
    assert tin in subgraph.inputs and tin not in subgraph.outputs
    assert pre_out not in (subgraph.inputs + subgraph.outputs)
    assert tout in subgraph.outputs and tout not in subgraph.inputs


def test_run_dual_output(dual_output_model, trf_pass):
    for op in dual_output_model.subgraphs[0].operators:
        assert not trf_pass.match(op)


@pytest.fixture(params=[('FLOAT32', [1, 10]),
                        ('INT8', [1, 10]),
                        ('INT16', [1, 3, 3, 8]),
                        ('INT16', [1, 5, 5, 3])])
def non_matching_model(request):
    t_type, t_shape = request.param
    model = XCOREModel()
    subgraph = model.create_subgraph()

    tin = subgraph.create_tensor('input', type_=t_type, shape=t_shape, isinput=True)
    tout = subgraph.create_tensor('output', tin.type, tin.shape, isoutput=True)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.ABS),
                             inputs=[tin], outputs=[tout])

    return model


def test_non_match(trf_pass, non_matching_model):
    for output_tensor in non_matching_model.subgraphs[0].outputs:
        assert not trf_pass.match(output_tensor)


if __name__ == "__main__":
    pytest.main()
