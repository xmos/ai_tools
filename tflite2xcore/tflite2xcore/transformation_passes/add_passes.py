# Copyright (c) 2020, XMOS Ltd, All rights reserved

from typing import Iterable
from copy import copy
import numpy as np

from tflite2xcore.xcore_model import Operator, Subgraph
from tflite2xcore.xcore_schema import (
    BuiltinOpCodes,
    OperatorCode,
    XCOREOpCodes,
    TensorType,
)

from .transformation_passes import ReplaceQuantizedOperatorPass


class ReplaceAddPass(ReplaceQuantizedOperatorPass):
    @property
    def matching_opcode(self) -> BuiltinOpCodes:
        return BuiltinOpCodes.ADD

    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_add_8)

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and len(op.inputs) == 2
            and op.inputs[0].type is self.matching_input_type
            and op.inputs[0].type is op.inputs[1].type is op.outputs[0].type
            and op.inputs[0].shape == op.inputs[1].shape == op.outputs[0].shape
        )

    def mutate(self, op: Operator):
        new_op = super().mutate(op)

        # passed variables
        s_0 = -6
        s_1 = s_0

        # calculate scale_mismatch
        scale0_scaleOut = (
            new_op.inputs[0].quantization["scale"][0]
            / new_op.outputs[0].quantization["scale"][0]
        )
        scale1_scaleOut = (
            new_op.inputs[1].quantization["scale"][0]
            / new_op.outputs[0].quantization["scale"][0]
        )

        n = max(scale0_scaleOut, scale1_scaleOut,)

        msb = 0
        n = int(n / 2)

        while n > 0:
            n = int(n / 2)
            msb += 1

        scale_mismatch = 14 - msb

        m_0 = scale0_scaleOut * 2 ** scale_mismatch
        m_1 = scale1_scaleOut * 2 ** scale_mismatch

        s_out = scale_mismatch + -s_0
        if s_out < 0:
            s_out = 0

        b = (
            (new_op.outputs[0].quantization["zero_point"][0] << s_out)
            - m_0 * (new_op.inputs[0].quantization["zero_point"][0] << -s_0)
            - m_1 * (new_op.inputs[1].quantization["zero_point"][0] << -s_1)
        )

        params = np.int32([s_0, m_0, s_1, m_1, b, s_out])

        subgraph = new_op.subgraph
        bias_scale_shift_tensor = subgraph.create_tensor(
            f"{new_op.name}/bias_scale_shift",
            TensorType.INT32,
            consumers=[new_op],
            shape=params.shape,
        )
        new_op.inputs.append(bias_scale_shift_tensor)

        new_op.inputs[2].buffer.data = params

        return new_op
