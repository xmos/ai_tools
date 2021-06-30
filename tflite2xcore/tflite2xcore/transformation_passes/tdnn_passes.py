# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import numpy as np

from tflite2xcore.transformation_passes.transformation_passes import (
    OperatorMatchingPass,
    QuantizedOperatorMatchingPass,
    TensorMatchingPass,
)
from tflite2xcore.xcore_model import Operator, Tensor
from tflite2xcore.xcore_schema import (
    OperatorCode,
    XCOREOpCodes,
    TensorType,
    BuiltinOpCodes,
)
from .pooling_passes import (
    ReplaceAveragePool2DPass,
    ReplaceGlobalAveragePool2DPass,
)


def insert_ringbuffer(ringbuffer_time_dim: int, new_op: Operator) -> Operator:
    ringbuffer_shape = list(new_op.inputs[0].shape)
    ringbuffer_shape[1] = ringbuffer_time_dim

    subgraph = new_op.subgraph

    ringbuffer_tensor = subgraph.create_tensor(
        f"{new_op.name}/ringbuffer",
        TensorType.INT8,
        consumers=[new_op],
        shape=ringbuffer_shape,
        quantization=new_op.inputs[0].quantization,
        custom_options={"tdnn":True},
    )

    old_data_shape = ringbuffer_shape
    old_data_shape[1] = old_data_shape[1] - 1
    old_data_tensor = subgraph.create_tensor(
        f"{new_op.name}/old_data",
        TensorType.INT8,
        shape=old_data_shape,
        custom_options={"tdnn":True},
    )

    persistent_buffer_number = subgraph.create_tensor(
        f"{new_op.name}/persistent_buffer_number", 
        TensorType.INT8,
        consumers=[new_op],
        shape=[1],
        custom_options={"tdnn":True},
    )
    breakpoint()
    #converts unique part of string name to int
    unique_part = new_op.name[new_op.name.find('_')+1:]
    print(unique_part)
    persistent_buffer_count = int(unique_part)

    # disconnect input from op
    new_op.inputs[0].consumers.pop(0)

    # create and connect ring buffer op
    subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_ringbuffer),
        inputs=[new_op.inputs[0], old_data_tensor, persistent_buffer_number],
        outputs=[ringbuffer_tensor],
        custom_options={"old_data_shape":np.prod(old_data_shape)},
    )
    # connect op to ring buffer
    new_op.inputs[0] = ringbuffer_tensor

    for input_tensor in new_op.inputs:
        input_tensor.add_custom_options(tdnn=True)
        
    new_op.inputs[2].buffer.data = persistent_buffer_count

    return new_op

class TdnnShallowinConv2dPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and "tdnn" not in op.custom_options
            and len(op.inputs[1].shape) >= 3
        )
     
    def mutate(self, op: Operator) -> Operator:
        op.add_custom_options(tdnn=True)

        # kernel_size[0]
        ringbuffer_time_dim = op.inputs[1].shape[1]

        new_op = insert_ringbuffer(ringbuffer_time_dim, op)

        return op

class TdnnMaxPool2DPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.MAX_POOL_2D

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and "tdnn" not in op.custom_options
        )
            
    def mutate(self, op: Operator) -> Operator:
        op.add_custom_options(tdnn=True)

        options = op.builtin_options

        ringbuffer_time_dim = options["filter_height"]

        op = insert_ringbuffer(ringbuffer_time_dim,op)
        
        return op

class TdnnAveragePool2DPass(ReplaceAveragePool2DPass):
    def mutate(self, op: Operator) -> Operator:
        new_op = super().mutate(op)

        ringbuffer_time_dim = new_op.custom_options["pool"][0]

        new_op = insert_ringbuffer(ringbuffer_time_dim, new_op)

        return new_op

class TdnnReshapePass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and op.operator_code.code is BuiltinOpCodes.RESHAPE
            and "tdnn" not in op.custom_options
        )

    def mutate(self, op: Operator) -> Operator:
        op.add_custom_options(tdnn=True)

        ringbuffer_time_dim = op.inputs[0].shape[1]

        op = insert_ringbuffer(ringbuffer_time_dim, op)
        
        new_op = super().mutate(op)

        return new_op 

class TdnnTensorPass(TensorMatchingPass):
    def match(self, tensor: Tensor) -> bool:
        return (
            super().match(tensor) 
            and "tdnn" not in tensor.custom_options
            #checks if tensor is 4d
            and len(tensor.shape) == 4
        )

    def mutate(self, tensor: Tensor) -> Tensor:
        tensor.add_custom_options(tdnn=True)

        shape = list(tensor.shape)
        shape[1] = 1
        tensor.shape = tuple(shape)

        return tensor
    
class TdnnCleanup(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and "tdnn" in op.custom_options
        )

    def mutate(self, op: Operator) -> bool:
        op.custom_options.pop('tdnn')
        return op

# class TdnnGlobalAveragePool2DPass(ReplaceGlobalAveragePool2DPass):
#     def mutate(self, op: Operator) -> Operator:
#         new_op = super().mutate(op)

#         ringbuffer_time_dim = new_op.inputs[0].shape[1]

#         new_op = insert_ringbuffer(ringbuffer_time_dim, new_op)

#         return new_op


# class TdnnGlobalMaxPool2DPass(OperatorMatchingPass):
#     @property
#     def matching_opcode(self) -> OperatorCode:
#         return BuiltinOpCodes.MAX

#     def match(self, op: Operator) -> bool:
#         return (
#             super().match(op)
#             and op.operator_code.code is self.matching_opcode
#             and "tdnn" not in op.custom_options
#         )

#     def mutate(self, op: Operator) -> Operator:
#         op.add_custom_options(tdnn=True)

#         ringbuffer_time_dim = op.inputs[0].shape[1]

#         op = insert_ringbuffer(ringbuffer_time_dim, op)

#         return op
