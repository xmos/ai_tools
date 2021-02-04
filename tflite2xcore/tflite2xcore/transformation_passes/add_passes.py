# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import numpy as np

from tflite2xcore.xcore_model import Operator
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

    def mutate(self, op: Operator) -> Operator:
        new_op = super().mutate(op)

        # constant picked so 8 bit number fits in 16 bits
        s_0 = s_1 = -6

        input_scales = (
            new_op.inputs[0].quantization["scale"][0],
            new_op.inputs[1].quantization["scale"][0],
        )

        output_scale = new_op.outputs[0].quantization["scale"][0]

        scale_ratios = (input_scales[0] / output_scale, input_scales[1] / output_scale)

        max_ratio = max(scale_ratios)

        msb_max_ratio = int(np.floor(np.log2(max_ratio)))

        # constant picked for number fits in 16 bits
        scale_mismatch = 14 - msb_max_ratio

        m_0 = np.round(scale_ratios[0] * 2 ** scale_mismatch)
        m_1 = np.round(scale_ratios[1] * 2 ** scale_mismatch)

        s_out = max(0, scale_mismatch - s_0)

        output_zero_point = new_op.outputs[0].quantization["zero_point"][0]

        inputs_zero_points = (
            new_op.inputs[0].quantization["zero_point"][0],
            new_op.inputs[1].quantization["zero_point"][0],
        )

        b = (
            (output_zero_point << s_out)
            - m_0 * (inputs_zero_points[0] << -s_0)
            - m_1 * (inputs_zero_points[1] << -s_1)
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
