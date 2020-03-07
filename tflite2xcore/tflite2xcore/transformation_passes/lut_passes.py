# Copyright (c) 2020, XMOS Ltd, All rights reserved
import numpy as np
from abc import abstractmethod
from tflite2xcore.operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from tflite2xcore.xcore_model import TensorType
from .transformation_passes import ReplaceQuantizedOperatorPass


class ReplaceWithXCLookup8Pass(ReplaceQuantizedOperatorPass):
    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_lookup_8)

    def _dequantize(self, int_arr):
        input_quant = self._input.quantization
        return (np.int32(int_arr) - input_quant['zero_point'][0]) * input_quant['scale'][0]

    def _quantize(self, float_arr):
        output_quant = self._output.quantization
        return np.int8(np.round(np.clip(
            float_arr / output_quant['scale'][0] + output_quant['zero_point'][0],
            -128, 127
        )))

    @abstractmethod
    def activation(self, float_arr):
        pass

    def mutate(self, op):
        new_op = super().mutate(op)

        inputs_int = np.arange(-128, 128, dtype=np.int8)
        with self.using(new_op):
            outputs_int = self._quantize(self.activation(self._dequantize(inputs_int)))
        outputs_int = np.concatenate([outputs_int[128:], outputs_int[0:128]])

        lut_tensor = new_op.subgraph.create_tensor(
            f"{op.name}/LUT", TensorType.INT8, shape=[len(outputs_int)],
            consumers=[new_op])
        lut_tensor.buffer.data = outputs_int
        new_op.inputs.append(lut_tensor)


class ReplaceReLUPass(ReplaceWithXCLookup8Pass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.RELU

    def activation(self, float_arr):
        return np.maximum(float_arr, 0.)


class ReplaceReLU6Pass(ReplaceWithXCLookup8Pass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.RELU6

    def activation(self, float_arr):
        return np.minimum(np.maximum(float_arr, 0.), 6.)


class ReplaceTanhPass(ReplaceWithXCLookup8Pass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.TANH

    def activation(self, float_arr):
        return np.tanh(float_arr)


class ReplaceLogisticPass(ReplaceWithXCLookup8Pass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.LOGISTIC

    def activation(self, float_arr):
        return 1. / (1. + np.exp(-float_arr))
