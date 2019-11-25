# Copyright (c) 2019, XMOS Ltd, All rights reserved

from GraphTransformer import PassPriority
from GraphTransformer import OperatorMatchingPass, InputTensorMatchingPass, OutputTensorMatchingPass
from OperatorCodes import BuiltinOpCodes, OperatorCode


class RemoveQuantizerFloatInputPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def match(self, op):
        if op.operator_code.code == BuiltinOpCodes.QUANTIZE:
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            if (input_tensor in op.subgraph.inputs
                    and output_tensor not in op.subgraph.outputs):
                if output_tensor.type == 'INT8' and input_tensor.type == 'FLOAT32':
                    return True

        return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.inputs.remove(op.inputs[0])
        subgraph.inputs.append(op.outputs[0])
        subgraph.tensors.remove(op.inputs[0])
        subgraph.operators.remove(op)


class RemoveDequantizerFloatOutputPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def match(self, op):
        if op.operator_code.code == BuiltinOpCodes.DEQUANTIZE:
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            if (output_tensor in op.subgraph.outputs
                    and input_tensor not in op.subgraph.inputs):
                if output_tensor.type == 'FLOAT32' and input_tensor.type == 'INT8':
                    return True

        return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.outputs.remove(op.outputs[0])
        subgraph.outputs.append(op.inputs[0])
        subgraph.tensors.remove(op.outputs[0])
        subgraph.operators.remove(op)


class AddQuantizerFloatInputPass(InputTensorMatchingPass):
    def __init__(self, priority=PassPriority.LOW):
        super().__init__(priority)

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


class AddDequantizerFloatOutputPass(OutputTensorMatchingPass):
    def __init__(self, priority=PassPriority.LOW):
        super().__init__(priority)

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


# TODO: implement tests
class RemoveOutputSoftmaxPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.MEDIUM):
        super().__init__(priority)

    def match(self, op):
        if op.operator_code.code == BuiltinOpCodes.SOFTMAX:
            if op.outputs[0] in op.subgraph.outputs:
                return True

        return False

    def mutate(self, op):
        pass
        # TODO: finish implementing me


# TODO: implement tests
class AddOutputArgmaxPass(OutputTensorMatchingPass):
    def __init__(self, priority=PassPriority.LOW):
        super().__init__(priority)

    def run_subgraph(self, subgraph):
        if len(subgraph.outputs) == 1:
            tensor = subgraph.outputs[0]
            if self.match(tensor):
                self.log_match(tensor)
                self.mutate(tensor)

    def match(self, tensor):
        return True  # TODO: consider some restriction on type/shape

    def mutate(self, tensor):
        pass
        # TODO: finish implementing me
