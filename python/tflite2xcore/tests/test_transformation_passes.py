# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from transformation_passes import RemoveQuantizerFloatInputPass
from transformation_passes import RemoveDequantizerFloatOutputPass
from GraphTransformer import PassManager
from OperatorCodes import OperatorCode, BuiltinOpCodes, XCOREOpCodes
from xcore_model import XCOREModel, Subgraph, Tensor, Operator, Buffer


class Test_RemoveQuantizerFloatInputPass():
    @pytest.fixture()
    def simple_model(self):
        model = XCOREModel()
        subgraph = model.create_subgraph()

        fin = subgraph.create_tensor('input', 'FLOAT32', [1, 5, 5, 3], isinput=True)
        qin = subgraph.create_tensor('quantized_input', 'INT8', fin.shape)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.QUANTIZE),
                                 inputs=[fin], outputs=[qin])

        qout = subgraph.create_tensor('quantized_output', 'INT8', qin.shape, isoutput=True)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.ABS),
                                 inputs=[qin], outputs=[qout])

        return model

    @pytest.fixture()
    def dual_input_model(self):
        model = XCOREModel()
        subgraph = model.create_subgraph()

        fin1 = subgraph.create_tensor('input_1', 'FLOAT32', [1, 5, 5, 3], isinput=True)
        qin1 = subgraph.create_tensor('quantized_input_1', 'INT8', fin1.shape)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.QUANTIZE),
                                 inputs=[fin1], outputs=[qin1])

        fin2 = subgraph.create_tensor('input_2', 'FLOAT32', fin1.shape, isinput=True)
        qin2 = subgraph.create_tensor('quantized_input_2', 'INT8', fin2.shape)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.QUANTIZE),
                                 inputs=[fin2], outputs=[qin2])

        qout = subgraph.create_tensor('quantized_output', 'INT8', qin1.shape, isoutput=True)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.ADD),
                                 inputs=[qin1, qin2], outputs=[qout])

        return model

    def non_matching_model(self):
        model = XCOREModel()
        subgraph = model.create_subgraph()

        fin1 = subgraph.create_tensor('input_1', 'FLOAT32', [1, 5, 5, 3], isinput=True)
        qout1 = subgraph.create_tensor('quantized_output_1', 'INT8', fin1.shape, isoutput=True)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.QUANTIZE),
                                 inputs=[fin1], outputs=[qout1])

        fin2 = subgraph.create_tensor('input_2', 'FLOAT32', [1, 3, 3, 8], isinput=True)
        qout2 = subgraph.create_tensor('quantized_output_2', 'INT8', fin2.shape, isoutput=True)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.QUANTIZE),
                                 inputs=[fin2], outputs=[qout2])

        return model

    @pytest.fixture()
    def trf_pass(self):
        return RemoveQuantizerFloatInputPass()

    def test_match(self, simple_model, trf_pass):
        assert trf_pass.match(simple_model.subgraphs[0].operators[0])

    def test_mutate(self, simple_model, trf_pass):
        subgraph = simple_model.subgraphs[0]
        trf_pass.mutate(subgraph.operators[0])

        assert len(subgraph.operators) == 1
        assert subgraph.operators[0].operator_code.code == BuiltinOpCodes.ABS
        assert len(subgraph.tensors) == 2

    def test_run_simple(self, simple_model, trf_pass):
        trf_pass.run(simple_model)
        subgraph = simple_model.subgraphs[0]

        assert len(subgraph.operators) == 1
        assert subgraph.operators[0].operator_code.code == BuiltinOpCodes.ABS
        assert len(subgraph.tensors) == 2

    def test_run_dual_input(self, dual_input_model, trf_pass):
        trf_pass.run(dual_input_model)
        subgraph = dual_input_model.subgraphs[0]

        assert len(subgraph.operators) == 1
        assert subgraph.operators[0].operator_code.code == BuiltinOpCodes.ADD
        assert len(subgraph.tensors) == 3

    def test_run_non_matching(self, trf_pass):
        non_matching_model = self.non_matching_model()
        trf_pass.run(non_matching_model)
        subgraph = non_matching_model.subgraphs[0]
        ref_subgraph = self.non_matching_model().subgraphs[0]

        assert len(subgraph.operators) == len(ref_subgraph.operators)
        for o1, o2 in zip(subgraph.operators, ref_subgraph.operators):
            assert o1.operator_code == o2.operator_code
        assert len(subgraph.tensors) == len(ref_subgraph.tensors)
        for t1, t2 in zip(subgraph.tensors, ref_subgraph.tensors):
            assert t1.type == t2.type
            assert t1.name == t2.name
            assert t1.shape == t2.shape


class Test_RemoveDequantizerFloatOutputPass():
    @pytest.fixture()
    def simple_model(self):
        model = XCOREModel()
        subgraph = model.create_subgraph()

        qin = subgraph.create_tensor('quantized_input', 'INT8', [1, 5, 5, 3], isinput=True)
        qout = subgraph.create_tensor('quantized_output', 'INT8', qin.shape)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.ABS),
                                 inputs=[qin], outputs=[qout])

        fout = subgraph.create_tensor('output', 'FLOAT32', qout.shape, isoutput=True)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.DEQUANTIZE),
                                 inputs=[qout], outputs=[fout])

        return model

    @pytest.fixture()
    def dual_output_model(self):
        model = XCOREModel()
        subgraph = model.create_subgraph()

        # TODO: add operator options to specify split axis and number
        qin = subgraph.create_tensor('quantized_input', 'INT8', [1, 5, 5, 4], isinput=True)
        qout1 = subgraph.create_tensor('quantized_output_1', 'INT8', [1, 5, 5, 2])
        qout2 = subgraph.create_tensor('quantized_output_2', 'INT8', [1, 5, 5, 2])
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.SPLIT),
                                 inputs=[qin], outputs=[qout1, qout2])

        fout1 = subgraph.create_tensor('output_1', 'FLOAT32', qout1.shape, isoutput=True)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.DEQUANTIZE),
                                 inputs=[qout1], outputs=[fout1])

        fout2 = subgraph.create_tensor('output_2', 'FLOAT32', qout1.shape, isoutput=True)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.DEQUANTIZE),
                                 inputs=[qout2], outputs=[fout2])

        return model

    def non_matching_model(self):
        model = XCOREModel()
        subgraph = model.create_subgraph()

        qin1 = subgraph.create_tensor('quantized_input_1', 'INT8', [1, 5, 5, 3], isinput=True)
        fout1 = subgraph.create_tensor('output_1', 'FLOAT32', qin1.shape, isoutput=True)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.DEQUANTIZE),
                                 inputs=[qin1], outputs=[fout1])

        qin2 = subgraph.create_tensor('quantized_input_2', 'INT8', [1, 3, 3, 8], isinput=True)
        fout2 = subgraph.create_tensor('output_2', 'FLOAT32', qin2.shape, isoutput=True)
        subgraph.create_operator(OperatorCode(BuiltinOpCodes.DEQUANTIZE),
                                 inputs=[qin2], outputs=[fout2])

        return model

    @pytest.fixture()
    def trf_pass(self):
        return RemoveDequantizerFloatOutputPass()

    def test_match(self, simple_model, trf_pass):
        assert trf_pass.match(simple_model.subgraphs[0].operators[1])

    def test_mutate(self, simple_model, trf_pass):
        subgraph = simple_model.subgraphs[0]
        trf_pass.mutate(subgraph.operators[1])

        assert len(subgraph.operators) == 1
        assert subgraph.operators[0].operator_code.code == BuiltinOpCodes.ABS
        assert len(subgraph.tensors) == 2

    def test_run_simple(self, simple_model, trf_pass):
        trf_pass.run(simple_model)
        subgraph = simple_model.subgraphs[0]

        assert len(subgraph.operators) == 1
        assert subgraph.operators[0].operator_code.code == BuiltinOpCodes.ABS
        assert len(subgraph.tensors) == 2

    def test_run_dual_output(self, dual_output_model, trf_pass):
        trf_pass.run(dual_output_model)
        subgraph = dual_output_model.subgraphs[0]

        assert len(subgraph.operators) == 1
        assert subgraph.operators[0].operator_code.code == BuiltinOpCodes.SPLIT
        assert len(subgraph.tensors) == 3

    def test_run_non_matching(self, trf_pass):
        non_matching_model = self.non_matching_model()
        trf_pass.run(non_matching_model)
        subgraph = non_matching_model.subgraphs[0]
        ref_subgraph = self.non_matching_model().subgraphs[0]

        assert len(subgraph.operators) == len(ref_subgraph.operators)
        for o1, o2 in zip(subgraph.operators, ref_subgraph.operators):
            assert o1.operator_code == o2.operator_code
        assert len(subgraph.tensors) == len(ref_subgraph.tensors)
        for t1, t2 in zip(subgraph.tensors, ref_subgraph.tensors):
            assert t1.type == t2.type
            assert t1.name == t2.name
            assert t1.shape == t2.shape


if __name__ == "__main__":
    pytest.main()
