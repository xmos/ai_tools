# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from .transformation_passes import (
    BufferMatchingPass,
    TensorMatchingPass,
    OperatorMatchingPass,
)


class EliminateDeadOperatorsPass(OperatorMatchingPass):
    def match(self, op):
        if super().match(op):
            interface_tensors = set(op.subgraph.inputs + op.subgraph.outputs)
            for t in op.outputs:
                if t in interface_tensors or t.consumers:
                    return False
            else:
                return True

        return False

    def mutate(self, op):
        op.subgraph.remove_operator(op)


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
        self.logger.debug(f"Removed {modified_cnt} dead buffers")
        return modified_cnt
