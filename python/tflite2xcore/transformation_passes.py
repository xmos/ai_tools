# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy as np

from abc import abstractmethod
from contextlib import contextmanager
from .graph_transformer import PassPriority
from .graph_transformer import OperatorMatchingPass, InputTensorMatchingPass, OutputTensorMatchingPass
from .operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from .xcore_model import TensorType


class RemoveQuantizerFloatInputPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def match(self, op):
        if op.operator_code.code == BuiltinOpCodes.QUANTIZE:
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            return (input_tensor in op.subgraph.inputs
                    and output_tensor not in op.subgraph.outputs
                    and output_tensor.type == TensorType.INT8
                    and input_tensor.type == TensorType.FLOAT32)

        return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.inputs.append(op.outputs[0])
        subgraph.remove_tensor(op.inputs[0])
        subgraph.remove_operator(op)


class RemoveDequantizerFloatOutputPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def match(self, op):
        if op.operator_code.code == BuiltinOpCodes.DEQUANTIZE:
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            return (output_tensor in op.subgraph.outputs
                    and input_tensor not in op.subgraph.inputs
                    and output_tensor.type == TensorType.FLOAT32
                    and input_tensor.type == TensorType.INT8)

        return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.outputs.append(op.inputs[0])
        subgraph.remove_tensor(op.outputs[0])
        subgraph.remove_operator(op)


class AddQuantizerFloatInputPass(InputTensorMatchingPass):
    def __init__(self, priority=PassPriority.LOW):
        super().__init__(priority)

    def match(self, input_tensor):
        return (input_tensor.type == TensorType.INT8)

    def mutate(self, qin):
        subgraph = qin.subgraph
        fin = subgraph.create_tensor(
            qin.name + '_float', TensorType.FLOAT32, qin.shape, isinput=True)
        subgraph.inputs.remove(qin)
        subgraph.inputs.append(fin)
        subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin], outputs=[qin])


class AddDequantizerFloatOutputPass(OutputTensorMatchingPass):
    def __init__(self, priority=PassPriority.LOW):
        super().__init__(priority)

    def match(self, input_tensor):
        return input_tensor.type == TensorType.INT8

    def mutate(self, qout):
        subgraph = qout.subgraph
        fout = subgraph.create_tensor(
            qout.name + '_float', TensorType.FLOAT32, qout.shape, isoutput=True)
        subgraph.outputs.remove(qout)
        subgraph.outputs.append(fout)
        subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qout], outputs=[fout])


class RemoveSoftmaxOutputPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.MEDIUM):
        super().__init__(priority)

    def match(self, op):
        return (op.operator_code.code == BuiltinOpCodes.SOFTMAX
                and op.outputs[0] in op.subgraph.outputs)

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.outputs.append(op.inputs[0])
        subgraph.remove_tensor(op.outputs[0])
        subgraph.remove_operator(op)


class AddArgmaxOutputPass(OutputTensorMatchingPass):
    def __init__(self, priority=PassPriority.LOW):
        super().__init__(priority)

    def match(self, tensor):
        return (len(tensor.subgraph.outputs) == 1
                and tensor.subgraph.outputs[0].type == TensorType.INT16
                and len(tensor.shape) == 2)

    def mutate(self, tensor):
        subgraph = tensor.subgraph
        tout = subgraph.create_tensor(
            tensor.name + '_argmax', tensor.type, tensor.shape, isoutput=True)
        subgraph.outputs.remove(tensor)
        subgraph.outputs.append(tout)
        subgraph.create_operator(
            OperatorCode(XCOREOpCodes.XC_argmax_16), inputs=[tensor], outputs=[tout])


class ReplaceQuantizedOperatorPass(OperatorMatchingPass):
    def __init__(self, priority):
        super().__init__(priority)
        self._op = None

    @contextmanager
    def using(self, op):
        self._op, original_op = op, self._op
        yield
        self._op = original_op

    @property
    def _output(self):
        return self._op.outputs[0]

    @property
    def _input(self):
        return self._op.inputs[0]

    @property
    def _weights(self):
        return self._op.inputs[1]

    @property
    def _biases(self):
        return self._op.inputs[2]

    @property
    def _multiplier(self):
        output_scale = self._output.quantization['scale'][0]
        bias_scale = np.array(self._biases.quantization['scale'])
        return bias_scale / output_scale

    @property
    def _unified_bias(self):
        weights = self._weights.numpy
        biases = self._biases.numpy
        input_zero_point = int(self._input.quantization['zero_point'][0])
        output_zero_point = int(self._output.quantization['zero_point'][0])

        zero_point_bias = np.sum(weights * input_zero_point,
                                 axis=tuple(j for j in range(1, len(weights.shape))))
        return np.int32(biases - zero_point_bias
                        + np.int(np.round(output_zero_point / self._multiplier)))

    @property
    def _shift_scale(self):
        multiplier = self._multiplier
        # NOTE: VLMUL expects one factor in Q2.14
        # we have 1 <= scale < 2 represented in Q2.14
        rshift = -np.ceil(np.log2(multiplier)) + 1
        scale = np.round(2**14 * (multiplier * 2**rshift))

        for j in range(len(scale)):
            if scale[j] == 2**15:
                rshift[j] -= 1
                scale[j] /= 2
            # we are using 16 bits instead of 8 so we need to adjust the shift
            # NOTE: VDEPTH8 shifts down by 8 bits, not 7 as stated on some pages of the ISA
            rshift[j] -= 8

        bias_size = self._biases.numpy.size
        if len(scale) == 1:
            rshift = np.repeat(rshift, bias_size)
            scale = np.repeat(scale, bias_size)
        return np.int16(rshift), np.int16(scale)

    def add_shift_scale(self, op):
        # calculate right shift/scale
        with self.using(op):
            rshift, scale = self._shift_scale
        if rshift.shape != scale.shape:
            raise ValueError(f"Shift and scale shapes don't match: {rshift.shape} != {scale.shape}")

        # add tensor and buffer for rshift/scale
        shift_scale_arr = np.hstack([rshift, scale]).reshape((2, -1))
        shift_scale_tensor = op.subgraph.create_tensor(
            f"{op.name}/shift_scale",
            TensorType.INT16,
            shift_scale_arr.shape,
            buffer=op.model.create_buffer(shift_scale_arr)
        )
        op.inputs.append(shift_scale_tensor)


class ReplaceDeepinShallowoutFullyConnectedOutputPass(ReplaceQuantizedOperatorPass):
    def __init__(self, priority=PassPriority.MEDIUM):
        super().__init__(priority)

    def match(self, op):
        if op.operator_code.code == BuiltinOpCodes.FULLY_CONNECTED:
            weight_tensor, output_tensor = op.inputs[1], op.outputs[0]
            return (output_tensor in op.subgraph.outputs
                    and weight_tensor.shape[0] < 16
                    and weight_tensor.shape[1] % 32 == 0)

        return False

    def mutate_output(self, op):
        with self.using(op):
            self._output.type = TensorType.INT16
            self._output.name = f"{op.name}/output"
            self._output.quantization = {
                'scale': [self._output.quantization['scale'][0] / 2**8],
                'zero_point': [int(self._output.quantization['zero_point'][0] * 2**8)],
                'details_type': "CustomQuantization",
                'quantized_dimension': 0
            }

    def mutate_weights(self, op):
        with self.using(op):
            # rename weight tensor
            # NOTE: no weight layout rearrangement is done for this op
            self._weights.name = f"{op.name}/weights"

    def mutate_biases(self, op):
        with self.using(op):
            # calculate and save a unified bias vector
            self._biases.buffer.data = self._unified_bias
            # rename bias tensor and change quantization mode to alert users to unusual layout
            self._biases.name = f"{op.name}/biases"
            self._biases.quantization['details_type'] = 'CustomQuantization'

    def replace_op(self, op):
        op.subgraph.create_operator(
            OperatorCode(XCOREOpCodes.XC_fc_deepin_shallowout_final),
            inputs=op.inputs,
            outputs=op.outputs
        )
        op.subgraph.remove_operator(op)

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        self.add_shift_scale(op)
        self.mutate_biases(op)
        self.mutate_weights(op)
        self.mutate_output(op)
        self.replace_op(op)
