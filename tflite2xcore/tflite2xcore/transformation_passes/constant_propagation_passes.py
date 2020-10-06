# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.xcore_schema import BuiltinOpCodes
from tflite2xcore.xcore_model import Operator

from .transformation_passes import OperatorMatchingPass


class ConstantPropagationPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        if super().match(op):
            for t in op.inputs:
                if not t.is_constant:
                    return False

            if op.operator_code.code in BuiltinOpCodes:
                return True
            else:
                self.logger.warning(
                    f"Found unsupported operator {op.operator_code.code}"
                )

        return False

    def mutate(self, op: Operator) -> None:
        raise NotImplementedError()  # TODO: finish this

