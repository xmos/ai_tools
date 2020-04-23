# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.xcore_schema import XCOREOpCodes
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    TensorMatchingPass,
    OperatorMatchingPass,
)


class RemoveUnusedBuffersPass(ModelTransformationPass):
    # TODO: modify this to use match/mutate
    def run(self, model):
        dangling = [j for j, b in enumerate(model.buffers) if not b.owners]
        if dangling:
            self.logger.debug(f"Found dangling buffers: {dangling}")

        model.buffers = [b for b in model.buffers if b.owners]
        if dangling:
            self.logger.info(f"Removed {len(dangling)} dangling buffers")

        return len(dangling)


class RemoveDanglingTensorsPass(TensorMatchingPass):
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
