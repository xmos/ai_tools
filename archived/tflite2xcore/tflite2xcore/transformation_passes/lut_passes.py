# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import numpy as np

from tflite2xcore.utils import quantize, dequantize
from tflite2xcore.xcore_schema import (
    TensorType,
    BuiltinOpCodes,
    OperatorCode,
    XCOREOpCodes,
)

from .transformation_passes import (
    ReplaceQuantizedOperatorPass,
    QuantizedOperatorMatchingPass,
)


ACTIVATIONS = {
    BuiltinOpCodes.RELU: lambda x: np.maximum(x, 0.0),
    BuiltinOpCodes.RELU6: lambda x: np.minimum(np.maximum(x, 0.0), 6.0),
    BuiltinOpCodes.TANH: lambda x: np.tanh(x),
    BuiltinOpCodes.LOGISTIC: lambda x: 1.0 / (1.0 + np.exp(-x)),
}


class ReplaceWithXCLookupPass(ReplaceQuantizedOperatorPass):
    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_lookup_8)

    def mutate(self, op):
        new_op = super().mutate(op)
        new_op.add_custom_options(original_opcode=self.matching_opcode)
        return new_op


class LegalizeXCLookupTablePass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_lookup_8

    def match(self, op):
        return super().match(op) and "original_opcode" in op.custom_options

    def _dequantize_input(self, int_arr):
        input_quant = self._input.quantization
        return dequantize(
            int_arr, input_quant["scale"][0], input_quant["zero_point"][0]
        )

    def _quantize_output(self, float_arr):
        output_quant = self._output.quantization
        return quantize(
            float_arr, output_quant["scale"][0], output_quant["zero_point"][0]
        )

    def mutate(self, op):
        inputs_int = np.arange(-128, 128, dtype=np.int8)
        activation = ACTIVATIONS[op.custom_options.pop("original_opcode")]
        with self.using(op):
            outputs_int = self._quantize_output(
                activation(self._dequantize_input(inputs_int))
            )
        outputs_int = np.concatenate([outputs_int[128:], outputs_int[0:128]])

        lut_tensor = op.subgraph.create_tensor(
            f"{op.name}/LUT", TensorType.INT8, shape=[len(outputs_int)], consumers=[op]
        )
        lut_tensor.buffer.data = outputs_int
        op.inputs.append(lut_tensor)


class ReplaceReLUPass(ReplaceWithXCLookupPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.RELU


class ReplaceReLU6Pass(ReplaceWithXCLookupPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.RELU6


class ReplaceTanhPass(ReplaceWithXCLookupPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.TANH


class ReplaceLogisticPass(ReplaceWithXCLookupPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.LOGISTIC
