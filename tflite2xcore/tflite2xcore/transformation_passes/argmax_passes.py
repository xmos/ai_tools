# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy

from tflite2xcore.operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from tflite2xcore.xcore_model import TensorType
from tflite2xcore.graph_transformer import OutputTensorMatchingPass, PassPriority
from .transformation_passes import ReplaceQuantizedOperatorPass


class AddArgMax16OutputPass(OutputTensorMatchingPass):
    def __init__(self, priority=PassPriority.ARGMAX):
        super().__init__(priority)

    def match(self, tensor):
        return (super().match(tensor)
                and len(tensor.subgraph.outputs) == 1
                and tensor.subgraph.outputs[0].type == TensorType.INT16
                and len(tensor.shape) == 2)

    def mutate(self, tensor):
        subgraph = tensor.subgraph
        tout = subgraph.create_tensor(
            f"{tensor.name}_argmax", TensorType.INT32, tensor.shape[:1], isoutput=True)
        subgraph.outputs.remove(tensor)
        op = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.ARG_MAX), inputs=[tensor], outputs=[tout])

        # add tensor with axis info
        dim_tensor = subgraph.create_tensor(
            f"{op.name}/axis", TensorType.INT32, shape=[],
            consumers=[op])
        op.inputs.append(dim_tensor)
        dim_tensor.buffer.data = numpy.int32([1])


class ReplaceArgMax16Pass(ReplaceQuantizedOperatorPass):
    def __init__(self, priority=PassPriority.ARGMAX):
        super().__init__(priority)

    @property
    def matching_opcode(self):
        return BuiltinOpCodes.ARG_MAX

    @property
    def _matching_input_type(self):
        return TensorType.INT16

    @property
    def _matching_output_type(self):
        return TensorType.INT32

    @property
    def _axis(self):
        return self._op.inputs[1].numpy

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_argmax_16)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (len(self._input.shape) == 2  # only 2D tensors are matched
                        and self._axis == 1)

    def mutate(self, op):
        new_op = super().mutate(op)
        new_op.subgraph.remove_tensor(new_op.inputs[1])
        new_op.inputs = new_op.inputs[:1]
        return new_op
