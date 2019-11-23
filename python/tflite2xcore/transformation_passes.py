# Copyright (c) 2019, XMOS Ltd, All rights reserved

from GraphTransformer import TransformationPass, PassPriority
from OperatorCodes import BuiltinOpCodes


class RemoveQuantizerFloatInputPass(TransformationPass):
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


class RemoveDequantizerFloatOutputPass(TransformationPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def match(self, op):
        # TODO: check that this is compliant with model data structure
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
