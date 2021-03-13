# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

from tflite2xcore.xcore_model import Operator
from tflite2xcore.xcore_schema import (
    OperatorCode,
    XCOREOpCodes,
    TensorType,
)

from .conv2d_passes import ReplaceDeepConv2dPass


class TdnnDeepConv2DPass(ReplaceDeepConv2dPass):
    def mutate(self, op: Operator):
        xc_conv_op = super().mutate(op)

        subgraph = xc_conv_op.subgraph

        ring_buffer_shape = list(xc_conv_op.inputs[0].shape)
        # kernel_size[0]
        ring_buffer_shape[1] = xc_conv_op.inputs[1].shape[1]
        ring_buffer_tensor = subgraph.create_tensor(
            f"{xc_conv_op.name}/ring_buffer",
            TensorType.INT8,
            consumers=[xc_conv_op],
            shape=ring_buffer_shape,
        )

        old_data_shape = ring_buffer_shape
        old_data_shape[1] = old_data_shape[1] - 1
        old_data_tensor = subgraph.create_tensor(
            f"{xc_conv_op.name}/old_data", TensorType.INT8, shape=old_data_shape
        )

        xc_conv_op.inputs[0].consumers.pop(0)

        subgraph.create_operator(
            OperatorCode(XCOREOpCodes.XC_ring_buffer),
            inputs=[xc_conv_op.inputs[0], old_data_tensor],
            outputs=[ring_buffer_tensor, old_data_tensor],
        )

        xc_conv_op.inputs[0] = ring_buffer_tensor

        return xc_conv_op
