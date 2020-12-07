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
        old_op = copy(op)
        new_op = super().mutate(op)

        # passed variables
        s_0 = -6
        s_1 = s_0

        # calculate scale_mismatch
        scale0_scaleOut = (
            old_op.inputs[0].quantization["scale"][0]
            / old_op.outputs[0].quantization["scale"][0]
        )
        scale1_scaleOut = (
            old_op.inputs[1].quantization["scale"][0]
            / old_op.outputs[0].quantization["scale"][0]
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
            (old_op.outputs[0].quantization["zero_point"][0] << s_out)
            - m_0 * (old_op.inputs[0].quantization["zero_point"][0] << -s_0)
            - m_1 * (old_op.inputs[1].quantization["zero_point"][0] << -s_1)
        )

        subgraph = new_op.subgraph
        bias_scale_shift_tensor = subgraph.create_tensor(
            f"{new_op.name}/bias_scale_shift",
            TensorType.INT32,
            consumers=[new_op],
            shape=[6],
        )
        new_op.inputs.append(bias_scale_shift_tensor)

        m_0 = np.array(m_0)
        m_1 = np.array(m_1)
        s_0 = np.array(s_0)
        s_1 = np.array(s_1)
        s_out = np.array(s_out)
        b = np.array(b)

        new_op.inputs[2].buffer.data = b"".join(
            # p.tostring() for p in self._bias_scale_shift
            p.tostring()
            for p in [
                b.astype(np.int32),
                m_0.astype(np.int32),
                m_1.astype(np.int32),
                s_0.astype(np.int32),
                s_1.astype(np.int32),
                s_out.astype(np.int32),
            ]
        )

        return new_op
