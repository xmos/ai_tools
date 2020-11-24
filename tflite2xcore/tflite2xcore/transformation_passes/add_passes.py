# Copyright (c) 2020, XMOS Ltd, All rights reserved

from typing import Iterable

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

    def mutate(self, op):
        new_op = super().mutate(op)
        subgraph = new_op.subgraph

        with self.using(new_op):
            # replace reduction_indices tensor with bias_scale_shift
            new_op.inputs[0].consumers.remove(new_op)
            new_op.inputs[0] = subgraph.create_tensor(
                f"{new_op.name}/bias_scale_shift",
                TensorType.INT8,
                shape=[7],
                consumers=[new_op],
            )

            # replace reduction_indices tensor with bias_scale_shift
            new_op.inputs[1].consumers.remove(new_op)
            new_op.inputs[1] = subgraph.create_tensor(
                f"{new_op.name}/bias_scale_shift",
                TensorType.INT8,
                shape=[7],
                consumers=[new_op],
            )

            # replace reduction_indices tensor with bias_scale_shift
            new_op.outputs[0].producers.remove(new_op)
            new_op.outputs[0] = subgraph.create_tensor(
                f"{new_op.name}/bias_scale_shift",
                TensorType.INT8,
                shape=[7],
                producers=[new_op],
            )

            # passed variables
            s_0 = 0
            s_1 = 0

            # calculate scale_mismatch
            scale0_scaleOut = (
                self._input.quantization["scale"][0]
                / self._output.quantization["scale"][0]
            )
            scale1_scaleOut = (
                self._input.quantization["scale"][1]
                / self._output.quantization["scale"][0]
            )

            n = max(
                scale0_scaleOut,
                scale1_scaleOut,
            )

            msb = 0
            n = int(n / 2)

            while n > 0:
                n = int(n / 2)
                msb += 1

            scale_mismatch = 16 - (msb + 1)

            m_0 = scale0_scaleOut << scale_mismatch
            m_1 = scale1_scaleOut << scale_mismatch

            b = (
                (self._output.quantization["zero_point"][0] << scale_mismatch)
                - m_0 * self._input.quantization["zero_point"][0]
                - m_1 * self._input.quantization["zero_point"][1]
            )

            s_out = scale_mismatch

            new_op.inputs[0].buffer.data = b"".join(
                # p.tostring() for p in self._bias_scale_shift
                p.tostring()
                for p in [m_0.astype(np.int16), s_0]
            )

            new_op.inputs[1].buffer.data = b"".join(
                # p.tostring() for p in self._bias_scale_shift
                p.tostring()
                for p in [m_1.astype(np.int16), s_1]
            )

            new_op.outputs[0].buffer.data = b"".join(
                # p.tostring() for p in self._bias_scale
                p.tostring()
                for p in [b.astype(np.int32), s_out]
            )

        return new_op
