# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from tflite2xcore.xcore_model import Operator, Tensor
from tflite2xcore.xcore_schema import (
    TensorType,
    BuiltinOpCodes,
    OperatorCode,
    ExternalOpCodes,
    ValidOpCodes,
)

from .transformation_passes import (
    InputTensorMatchingPass,
    OutputTensorMatchingPass,
    QuantizedOperatorMatchingPass,
)


class RemoveRedundantInt8RequantizationPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self) -> BuiltinOpCodes:
        return BuiltinOpCodes.QUANTIZE

    _PRECEDING_OPCODES = (
        BuiltinOpCodes.CONV_2D,
        BuiltinOpCodes.DEPTHWISE_CONV_2D,
        BuiltinOpCodes.FULLY_CONNECTED,
        BuiltinOpCodes.QUANTIZE,
    )

    def match(self, op: Operator) -> bool:
        if super().match(op):
            intermediate_tensor = op.inputs[0]
            if (
                len(intermediate_tensor.consumers) == 1
                and len(intermediate_tensor.producers) == 1
                and intermediate_tensor not in op.subgraph.outputs
            ):
                producer_op = intermediate_tensor.producers[0]
                return producer_op.operator_code.code in self._PRECEDING_OPCODES
        return False

    def mutate(self, op: Operator) -> None:
        subgraph = op.subgraph
        intermediate_tensor = op.inputs[0]
        output_tensor = op.outputs[0]

        producer_op = intermediate_tensor.producers[0]
        producer_op.outputs[0] = output_tensor
        output_tensor.producers.append(producer_op)

        # it is safe to remove the tensor and the intermediate op
        # since the match checked that there are no other consumers
        subgraph.remove_operator(op)
        subgraph.remove_tensor(intermediate_tensor)


# TODO: improve tests for this
class CanonicalizeQuantizedInputPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self) -> BuiltinOpCodes:
        return BuiltinOpCodes.QUANTIZE

    @property
    def matching_input_type(self) -> TensorType:
        return TensorType.FLOAT32

    def match(self, op: Operator) -> bool:
        if super().match(op):
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            return (
                input_tensor in op.subgraph.inputs
                and len(input_tensor.consumers) == 1
                and output_tensor not in op.subgraph.outputs
            )

        return False

    def mutate(self, op: Operator) -> None:
        subgraph = op.subgraph
        subgraph.inputs.append(op.outputs[0])
        subgraph.remove_tensor(op.inputs[0])  # DCE doesn't clean up subgraph inputs
        subgraph.remove_operator(op)


# TODO consider adding tests for this
class CanonicalizeLceQuantizedInputPass(CanonicalizeQuantizedInputPass):
    @property
    def matching_input_type(self) -> TensorType:
        return TensorType.INT8

    @property
    def matching_output_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def matching_opcode(self) -> ValidOpCodes:
        return ExternalOpCodes.LceQuantize


class CanonicalizeQuantizedOutputPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self) -> BuiltinOpCodes:
        return BuiltinOpCodes.DEQUANTIZE

    @property
    def matching_output_type(self) -> TensorType:
        return TensorType.FLOAT32

    def match(self, op: Operator) -> bool:
        if super().match(op):
            try:
                if op.operator_code.code is not self.matching_opcode:
                    return False
            except AttributeError:
                return False

            output_tensor = op.outputs[0]
            if (
                output_tensor in op.subgraph.outputs
                and not output_tensor.consumers
                and op.inputs[0] not in op.subgraph.inputs
            ):
                if len(output_tensor.producers) == 1:
                    return True
                else:
                    self.logger.warning(
                        f"Encountered output of removable {self.matching_opcode} "
                        "with more than one producer."
                    )

        return False

    def mutate(self, op: Operator) -> None:
        subgraph = op.subgraph
        subgraph.outputs.append(op.inputs[0])
        subgraph.remove_tensor(op.outputs[0])  # DCE doesn't clean up subgraph outputs
        subgraph.remove_operator(op)


# TODO consider adding tests for this
class CanonicalizeLceQuantizedOutputPass(CanonicalizeQuantizedOutputPass):
    @property
    def matching_input_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def matching_opcode(self) -> ValidOpCodes:
        return ExternalOpCodes.LceDequantize


# TODO: improve tests for this
class LegalizeFloatInputPass(InputTensorMatchingPass):
    def match(self, input_tensor: Tensor) -> bool:
        return super().match(input_tensor) and input_tensor.type is TensorType.INT8

    def mutate(self, qin: Tensor) -> None:
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
    def match(self, input_tensor: Tensor) -> bool:
        return super().match(input_tensor) and input_tensor.type is TensorType.INT8

    def mutate(self, qout: Tensor) -> None:
        subgraph = qout.subgraph
        fout = subgraph.create_tensor(
            f"{qout.name}_float", TensorType.FLOAT32, qout.shape, isoutput=True
        )
        subgraph.outputs.remove(qout)
        subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qout], outputs=[fout]
        )
