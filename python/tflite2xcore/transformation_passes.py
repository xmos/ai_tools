# Copyright (c) 2019, XMOS Ltd, All rights reserved

from graph_transformer import PassPriority
from graph_transformer import OperatorMatchingPass, InputTensorMatchingPass, OutputTensorMatchingPass
from operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes


class RemoveQuantizerFloatInputPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def match(self, op):
        if op.operator_code.code == BuiltinOpCodes.QUANTIZE:
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            if (input_tensor in op.subgraph.inputs
                    and output_tensor not in op.subgraph.outputs):
                return output_tensor.type == 'INT8' and input_tensor.type == 'FLOAT32'

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
                return output_tensor.type == 'FLOAT32' and input_tensor.type == 'INT8'

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
        return (input_tensor.type == 'INT8')

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
        return input_tensor.type == 'INT8'

    def mutate(self, qout):
        subgraph = qout.subgraph
        fout = subgraph.create_tensor(
            qout.name + '_float', 'FLOAT32', qout.shape, isoutput=True)
        subgraph.outputs.remove(qout)
        subgraph.outputs.append(fout)
        subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qout], outputs=[fout])


class RemoveSoftmaxOutputPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.MEDIUM):
        super().__init__(priority)

    def match(self, op):
        return (op.operator_code.code == BuiltinOpCodes.SOFTMAX
                and op.outputs[0] in op.subgraph.outputs)

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.outputs.remove(op.outputs[0])
        subgraph.outputs.append(op.inputs[0])
        subgraph.tensors.remove(op.outputs[0])
        subgraph.operators.remove(op)


class AddArgmaxOutputPass(OutputTensorMatchingPass):
    def __init__(self, priority=PassPriority.LOW):
        super().__init__(priority)

    def match(self, tensor):
        return (len(tensor.subgraph.outputs) == 1
                and tensor.subgraph.outputs[0].type == 'INT16'
                and len(tensor.shape) == 2)

    def mutate(self, tensor):
        subgraph = tensor.subgraph
        tout = subgraph.create_tensor(
            tensor.name + '_argmax', tensor.type, tensor.shape, isoutput=True)
        subgraph.outputs.remove(tensor)
        subgraph.outputs.append(tout)
        subgraph.create_operator(
            OperatorCode(XCOREOpCodes.XC_argmax_16), inputs=[tensor], outputs=[tout])


class ReplaceDeepinShallowoutFullyConnectedOutput(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.MEDIUM):
        super().__init__(priority)

    def match(self, op):
        if op.operator_code.code == BuiltinOpCodes.FULLY_CONNECTED:
            weight_tensor = op.inputs[1]
            return weight_tensor.shape[0] < 16 and weight_tensor.shape[1] % 32 == 0

        return False

    def mutate(self, op):
        pass
        # TODO: implement me
