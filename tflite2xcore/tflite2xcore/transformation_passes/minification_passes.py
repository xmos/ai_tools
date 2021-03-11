# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from tflite2xcore.xcore_schema import XCOREOpCodes, BuiltinOpCodes, Buffer, XCOREModel
from .transformation_passes import TensorMatchingPass, BufferMatchingPass


class MinifyQuantInfoPass(TensorMatchingPass):
    # NOTE: it's risky to include the builtin ops here, but (at least in the
    #       micro interpreter), min/max info does not seem to be used
    SAFE_OP_CODES = [c for c in XCOREOpCodes] + [c for c in BuiltinOpCodes]

    def match(self, tensor):
        dependents = tensor.consumers + tensor.producers
        quantization = tensor.quantization

        if super().match(tensor) and quantization and dependents:
            for op in dependents:
                if op.operator_code.code not in self.SAFE_OP_CODES:
                    # min/max info is removed if tensor only interacts with XC ops
                    return False
            else:
                return "min" in quantization or "max" in quantization
        return False

    def mutate(self, tensor):
        tensor.quantization.pop("min", None)
        tensor.quantization.pop("max", None)


class MinifyTensorNamesPass(TensorMatchingPass):
    def __new_tensor_name(self, tensor):
        return str(self._obj_index)

    def match(self, tensor):
        return super().match(tensor) and tensor.name != self.__new_tensor_name(tensor)

    def mutate(self, tensor):
        tensor.name = self.__new_tensor_name(tensor)


# TODO: add tests
class UnifyEmptyBuffersPass(BufferMatchingPass):
    def match(self, buffer: Buffer) -> bool:
        return (
            super().match(buffer)
            and not buffer
            and buffer is not buffer.model.buffers[0]
            and buffer.owners
        )

    def mutate(self, buffer: Buffer) -> None:
        sentinel = buffer.model.buffers[0]

        for owner in buffer.owners:
            owner.buffer = sentinel
            sentinel.owners.append(owner)

        buffer.owners = []

    def run(self, model: XCOREModel) -> int:
        model.buffers.insert(0, Buffer())
        modified_cnt = super().run(model)
        self.logger.debug(f"Unified {modified_cnt} empty buffers")
        return modified_cnt
