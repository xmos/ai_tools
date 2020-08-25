# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.xcore_schema import TensorType, BuiltinOpCodes, OperatorCode

from .transformation_passes import (
    OperatorMatchingPass,
    InputTensorMatchingPass,
    OutputTensorMatchingPass,
)

# TODO: improve tests for this
class CanonicalizeQuantizedInputPass(OperatorMatchingPass):
    def match(self, op):
        if super().match(op) and op.operator_code.code is BuiltinOpCodes.QUANTIZE:
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            return (
                input_tensor in op.subgraph.inputs
                and len(input_tensor.consumers) == 1
                and output_tensor not in op.subgraph.outputs
                and output_tensor.type is TensorType.INT8
                and input_tensor.type is TensorType.FLOAT32
            )

        return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.inputs.append(op.outputs[0])
        subgraph.remove_tensor(op.inputs[0])  # DCE doesn't clean up subgraph inputs
        subgraph.remove_operator(op)


class CanonicalizeQuantizedOutputPass(OperatorMatchingPass):
    def match(self, op):
        if super().match(op) and op.operator_code.code is BuiltinOpCodes.DEQUANTIZE:
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            if (
                output_tensor in op.subgraph.outputs
                and not output_tensor.consumers
                and input_tensor not in op.subgraph.inputs
                and output_tensor.type is TensorType.FLOAT32
                and input_tensor.type is TensorType.INT8
            ):
                if len(output_tensor.producers) == 1:
                    return True
                else:
                    self.logger.warning(
                        "Encountered output of removable DEQUANTIZE "
                        "with more than one producer."
                    )

        return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.outputs.append(op.inputs[0])
        subgraph.remove_tensor(op.outputs[0])  # DCE doesn't clean up subgraph outputs
        subgraph.remove_operator(op)


# TODO: improve tests for this
class LegalizeFloatInputPass(InputTensorMatchingPass):
    def match(self, input_tensor):
        return super().match(input_tensor) and input_tensor.type is TensorType.INT8

    def mutate(self, qin):
        subgraph = qin.subgraph
        fin = subgraph.create_tensor(
            f"{qin.name}_float", TensorType.FLOAT32, qin.shape, isinput=True
        )
        subgraph.inputs.remove(qin)
        op = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin], outputs=[qin]
        )
        # builtin interpreter prefers ops ordered this way
        subgraph.operators.remove(op)
        subgraph.operators.insert(0, op)


# TODO: improve tests for this
class LegalizeFloatOutputPass(OutputTensorMatchingPass):
    def match(self, input_tensor):
        return super().match(input_tensor) and input_tensor.type is TensorType.INT8

    def mutate(self, qout):
        subgraph = qout.subgraph
        fout = subgraph.create_tensor(
            f"{qout.name}_float", TensorType.FLOAT32, qout.shape, isoutput=True
        )
        subgraph.outputs.remove(qout)
        subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qout], outputs=[fout]
        )
