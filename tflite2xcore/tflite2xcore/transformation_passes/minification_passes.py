# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.operator_codes import XCOREOpCodes, BuiltinOpCodes
from .transformation_passes import TensorMatchingPass


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
                return 'min' in quantization or 'max' in quantization
        return False

    def mutate(self, tensor):
        tensor.quantization.pop('min', None)
        tensor.quantization.pop('max', None)


class MinifyTensorNamesPass(TensorMatchingPass):
    def __new_tensor_name(self, tensor):
        return str(self._obj_index)

    def match(self, tensor):
        return (super().match(tensor)
                and tensor.name != self.__new_tensor_name(tensor))

    def mutate(self, tensor):
        tensor.name = self.__new_tensor_name(tensor)
