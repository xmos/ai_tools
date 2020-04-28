# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.xcore_schema import XCOREOpCodes
from tflite2xcore.transformation_passes import (
    BufferMatchingPass,
    TensorMatchingPass,
    OperatorMatchingPass,
)


class EliminateDeadTensorsPass(TensorMatchingPass):
    def match(self, tensor):
        return (
            super().match(tensor)
            and tensor not in tensor.subgraph.inputs
            and tensor not in tensor.subgraph.outputs
            and not tensor.consumers
            and not tensor.producers
        )

    def mutate(self, tensor):
        tensor.subgraph.remove_tensor(tensor)


class EliminateDeadBuffersPass(BufferMatchingPass):
    def match(self, buffer):
        return super().match(buffer) and not buffer.owners

    def mutate(self, buffer):
        buffer.model.buffers.remove(buffer)

    def run(self, model):
        modified_cnt = super().run(model)
        self.logger.info(f"Removed {modified_cnt} dead buffers")
        return modified_cnt
