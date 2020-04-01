# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.graph_transformer import PassPriority
from tflite2xcore.graph_transformer import (
    ModelTransformationPass,
    TensorMatchingPass
)


class RemoveUnusedBuffersPass(ModelTransformationPass):
    def __init__(self, priority=PassPriority.CLEANUP):
        super().__init__(priority)

    def run(self, model):
        cnt_before = len(model.buffers)
        model.buffers = [b for b in model.buffers if b.owners]
        cnt_removed = cnt_before - len(model.buffers)
        if cnt_removed:
            self.logger.info(f"Removed {cnt_removed} dangling buffers")


class RemoveDanglingTensorsPass(TensorMatchingPass):
    def __init__(self, priority=PassPriority.CLEANUP):
        super().__init__(priority)

    def match(self, tensor):
        return (super().match(tensor)
                and tensor not in tensor.subgraph.inputs
                and tensor not in tensor.subgraph.outputs
                and not tensor.consumers
                and not tensor.producers)

    def mutate(self, tensor):
        tensor.subgraph.remove_tensor(tensor)