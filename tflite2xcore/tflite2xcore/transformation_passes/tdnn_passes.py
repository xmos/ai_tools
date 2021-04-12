# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

from tflite2xcore.transformation_passes.transformation_passes import (
    OperatorMatchingPass,
    TensorMatchingPass,
)
from tflite2xcore.xcore_model import Operator
from tflite2xcore.xcore_schema import (
    OperatorCode,
    XCOREOpCodes,
    TensorType,
)

from tflite2xcore.xcore_schema.op_codes import BuiltinOpCodes

from .pooling_passes import (
    ReplaceMaxPool2DPass,
    ReplaceAveragePool2DPass,
    ReplaceGlobalAveragePool2DPass,
)
from .conv2d_passes import ReplaceDeepConv2dPass


def insert_ring_buffer(ring_buffer_time_dim, new_op: Operator):
    ring_buffer_shape = list(new_op.inputs[0].shape)
    ring_buffer_shape[1] = ring_buffer_time_dim

    subgraph = new_op.subgraph

    ring_buffer_tensor = subgraph.create_tensor(
        f"{new_op.name}/ring_buffer",
        TensorType.INT8,
        consumers=[new_op],
        shape=ring_buffer_shape,
        custom_options="tdnn",
    )

    old_data_shape = ring_buffer_shape
    old_data_shape[1] = old_data_shape[1] - 1
    old_data_tensor = subgraph.create_tensor(
        f"{new_op.name}/old_data",
        TensorType.INT8,
        shape=old_data_shape,
        custom_options="tdnn",
    )

    new_op.inputs[0].consumers.pop(0)

    subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_ring_buffer),
        inputs=[new_op.inputs[0], old_data_tensor],
        outputs=[ring_buffer_tensor, old_data_tensor],
    )

    new_op.inputs[0] = ring_buffer_tensor

    return new_op


class TdnnDeepConv2dPass(ReplaceDeepConv2dPass):
    def mutate(self, op: Operator):
        new_op = super().mutate(op)

        # kernel_size[0]
        ring_buffer_time_dim = new_op.inputs[1].shape[1]

        new_op = insert_ring_buffer(ring_buffer_time_dim, new_op)

        return new_op


class TdnnMaxPool2DPass(ReplaceMaxPool2DPass):
    def mutate(self, op: Operator):
        new_op = super().mutate(op)

        ring_buffer_time_dim = new_op.custom_options["pool"][0]

        new_op = insert_ring_buffer(ring_buffer_time_dim, new_op)
        
        assert new_op.inputs[0].name == f"{new_op.name}/ring_buffer"
        assert 'XC_ring_buffer' in new_op.inputs[0].producers[0].name

        return new_op


class TdnnAveragePool2DPass(ReplaceAveragePool2DPass):
    def mutate(self, op: Operator):
        new_op = super().mutate(op)

        ring_buffer_time_dim = new_op.custom_options["pool"][0]

        new_op = insert_ring_buffer(ring_buffer_time_dim, new_op)

        return new_op


class TdnnGlobalAveragePool2DPass(ReplaceGlobalAveragePool2DPass):
    def mutate(self, op: Operator):
        new_op = super().mutate(op)

        ring_buffer_time_dim = new_op.inputs[0].shape[1]

        new_op = insert_ring_buffer(ring_buffer_time_dim, new_op)

        return new_op


class TdnnGlobalMaxPool2DPass(OperatorMatchingPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.MAX

    def match(self, op):
        return (
            super().match(op)
            and op.operator_code.code is self.matching_opcode
            and "tdnn" not in op.custom_options
        )

    def mutate(self, op: Operator):
        op.add_custom_options(tdnn=True)

        ring_buffer_time_dim = op.inputs[0].shape[1]

        op = insert_ring_buffer(ring_buffer_time_dim, op)

        return op


class TdnnFlattenPass(OperatorMatchingPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.FLATTEN

    def match(self, op):
        return (
            super().match(op)
            and op.operator_code.code is self.matching_opcode
            and "tdnn" not in op.custom_options
        )

    def mutate(self, op: Operator):
        op.add_custom_options(tdnn=True)

        ring_buffer_time_dim = op.inputs[0].shape[1]

        op = insert_ring_buffer(ring_buffer_time_dim, op)

        return op


class TdnnTensorPass(TensorMatchingPass):
    def match(self, tensor):
        return super().match(tensor) and "tdnn" not in tensor.custom_options

    def mutate(self, tensor):
        tensor.add_custom_options(tdnn=True)

        if len(tensor.shape) > 2:
            shape = list(tensor.shape)
            shape[1] = 1
            tensor.shape = shape

        return tensor

