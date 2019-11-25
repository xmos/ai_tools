# Copyright (c) 2019, XMOS Ltd, All rights reserved

from GraphTransformer import TransformationPass, PassPriority
from OperatorCodes import BuiltinOpCodes, OperatorCode


class RemoveQuantizerFloatInputPass(TransformationPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.operators

    def match(self, op):
        if op.operator_code.code == BuiltinOpCodes.QUANTIZE:
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            if (input_tensor in op.subgraph.inputs
                    and output_tensor not in op.subgraph.outputs):
                if output_tensor.type == 'INT8' and input_tensor.type == 'FLOAT32':
                    self.log_match(op)
                    return True

        return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.inputs.remove(op.inputs[0])
        subgraph.inputs.append(op.outputs[0])
        subgraph.tensors.remove(op.inputs[0])
        subgraph.operators.remove(op)


class RemoveDequantizerFloatOutputPass(TransformationPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.operators

    def match(self, op):
        if op.operator_code.code == BuiltinOpCodes.DEQUANTIZE:
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            if (output_tensor in op.subgraph.outputs
                    and input_tensor not in op.subgraph.inputs):
                if output_tensor.type == 'FLOAT32' and input_tensor.type == 'INT8':
                    self.log_match(op)
                    return True

        return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.outputs.remove(op.outputs[0])
        subgraph.outputs.append(op.inputs[0])
        subgraph.tensors.remove(op.outputs[0])
        subgraph.operators.remove(op)


class AddQuantizerFloatInputPass(TransformationPass):
    def __init__(self, priority=PassPriority.LOW):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.inputs

    def match(self, input_tensor):
        if input_tensor.type == 'INT8':
            return True

        return False

    def mutate(self, qin):
        subgraph = qin.subgraph
        fin = subgraph.create_tensor(
            qin.name + '_float', 'FLOAT32', qin.shape, isinput=True)
        subgraph.inputs.remove(qin)
        subgraph.inputs.append(fin)
        subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin], outputs=[qin])


class AddDequantizerFloatOutputPass(TransformationPass):
    def __init__(self, priority=PassPriority.LOW):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.outputs

    def match(self, input_tensor):
        if input_tensor.type in 'INT8':
            return True

        return False

    def mutate(self, qout):
        subgraph = qout.subgraph
        fout = subgraph.create_tensor(
            qout.name + '_float', 'FLOAT32', qout.shape, isoutput=True)
        subgraph.outputs.remove(qout)
        subgraph.outputs.append(fout)
        subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qout], outputs=[fout])
