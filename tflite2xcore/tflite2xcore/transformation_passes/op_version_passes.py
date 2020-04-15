# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.xcore_model import TensorType
from tflite2xcore.operator_codes import BuiltinOpCodes, OperatorCode
from tflite2xcore.transformation_passes import OperatorMatchingPass


# TODO: implement tests for this
class LegalizeQuantizeVersionPass(OperatorMatchingPass):
    def match(self, op):
        if super().match(op):
            opcode = op.operator_code
            return (
                opcode.code is BuiltinOpCodes.QUANTIZE
                and opcode.version == 2
                and op.inputs[0].type is TensorType.FLOAT32
                and op.outputs[0].type is TensorType.INT8
            )

    def mutate(self, op):
        op.operator_code = OperatorCode(BuiltinOpCodes.QUANTIZE, version=1)
