# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.xcore_model import Operator

from tflite2xcore.xcore_schema import TensorType, BuiltinOpCodes
from tflite2xcore.transformation_passes import OperatorMatchingPass


class LegalizeQuantizeVersionPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        if not super().match(op):
            return False

        opcode = op.operator_code
        return (
            opcode.code is BuiltinOpCodes.QUANTIZE
            and opcode.version == 2
            and op.inputs[0].type is TensorType.FLOAT32
            and op.outputs[0].type is TensorType.INT8
        )

    def mutate(self, op: Operator) -> None:
        op.operator_code.version = 1
