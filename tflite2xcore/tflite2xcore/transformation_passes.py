# Copyright (c) 2019, XMOS Ltd, All rights reserved

import logging
import numpy as np

from abc import abstractmethod
from contextlib import contextmanager
from tflite2xcore.graph_transformer import PassPriority
from tflite2xcore.graph_transformer import (
    ModelTransformationPass,
    OperatorMatchingPass,
    InputTensorMatchingPass,
    OutputTensorMatchingPass
)
from tflite2xcore.operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from tflite2xcore.xcore_model import TensorType


class RemoveQuantizerFloatInputPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.PREP):
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
    def __init__(self, priority=PassPriority.PREP):
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
    def __init__(self, priority=PassPriority.CLEANUP):
        super().__init__(priority)

    def match(self, input_tensor):
        return (input_tensor.type == TensorType.INT8)

    def mutate(self, qin):
        subgraph = qin.subgraph
        fin = subgraph.create_tensor(
            f"{qin.name}_float", TensorType.FLOAT32, qin.shape, isinput=True)
        subgraph.inputs.remove(qin)
        op = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin], outputs=[qin])
        # python interpreter prefers ops ordered this way
        subgraph.operators.remove(op)
        subgraph.operators.insert(0, op)


class AddDequantizerFloatOutputPass(OutputTensorMatchingPass):
    def __init__(self, priority=PassPriority.CLEANUP):
        super().__init__(priority)

    def match(self, input_tensor):
        return input_tensor.type == TensorType.INT8

    def mutate(self, qout):
        subgraph = qout.subgraph
        fout = subgraph.create_tensor(
            f"{qout.name}_float", TensorType.FLOAT32, qout.shape, isoutput=True)
        subgraph.outputs.remove(qout)
        subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.DEQUANTIZE), inputs=[qout], outputs=[fout])


class RemoveSoftmaxOutputPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def match(self, op):
        return (op.operator_code.code == BuiltinOpCodes.SOFTMAX
                and op.outputs[0] in op.subgraph.outputs)

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.outputs.append(op.inputs[0])
        subgraph.remove_tensor(op.outputs[0])
        subgraph.remove_operator(op)


class AddArgMax16OutputPass(OutputTensorMatchingPass):
    def __init__(self, priority=PassPriority.LOW):
        super().__init__(priority)

    def match(self, tensor):
        return (len(tensor.subgraph.outputs) == 1
                and tensor.subgraph.outputs[0].type == TensorType.INT16
                and len(tensor.shape) == 2)

    def mutate(self, tensor):
        subgraph = tensor.subgraph
        tout = subgraph.create_tensor(
            f"{tensor.name}_argmax", TensorType.INT32, tensor.shape[:1], isoutput=True)
        subgraph.outputs.remove(tensor)
        op = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.ARG_MAX), inputs=[tensor], outputs=[tout])

        # add tensor with axis info
        dim_tensor = subgraph.create_tensor(
            f"{op.name}/axis", TensorType.INT32, shape=[])
        op.inputs.append(dim_tensor)
        dim_tensor.buffer.data = np.int32([1])


# TODO: write (at least regression) tests for this class
class ReplaceQuantizedOperatorPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.MEDIUM):
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
    @abstractmethod
    def matching_opcode(self):
        raise NotImplementedError()

    @property
    def _matching_input_type(self):
        return TensorType.INT8

    @property
    def _matching_output_type(self):
        return TensorType.INT8

    @property
    @abstractmethod
    def new_opcode(self):
        raise NotImplementedError()

    def mutate(self, op):
        new_op = op.subgraph.create_operator(
            self.new_opcode, inputs=op.inputs, outputs=op.outputs)
        new_op.subgraph.replace_operator(op, new_op)
        return new_op

    def match(self, op):
        if op.operator_code.code == self.matching_opcode:
            with self.using(op):
                return (self._input.type == self._matching_input_type
                        and self._output.type == self._matching_output_type)


class ReplaceArgMax16Pass(ReplaceQuantizedOperatorPass):
    def __init__(self, priority=PassPriority.LOW):
        super().__init__(priority)

    @property
    def matching_opcode(self):
        return BuiltinOpCodes.ARG_MAX

    @property
    def _matching_input_type(self):
        return TensorType.INT16

    @property
    def _matching_output_type(self):
        return TensorType.INT32

    @property
    def _axis(self):
        return self._op.inputs[1].numpy

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_argmax_16)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (len(self._input.shape) == 2  # only 2D tensors are matched
                        and self._axis == 1)

    def mutate(self, op):
        new_op = super().mutate(op)
        new_op.subgraph.remove_tensor(new_op.inputs[1])
        new_op.inputs = new_op.inputs[:1]
        return new_op


# TODO: write (at least regression) tests for this class
class ReplaceXCOREWeightBiasOperatorPass(ReplaceQuantizedOperatorPass):
    @property
    def _weights(self):
        return self._op.inputs[1]

    @property
    def _biases(self):
        return self._op.inputs[2]

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._weights.type == TensorType.INT8
                        and self._biases.type == TensorType.INT32)

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
                        + np.int32(np.round(output_zero_point / self._multiplier)))

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
        rshift, scale = np.int16(rshift), np.int16(scale)
        if rshift.shape != scale.shape:
            raise ValueError(f"Shift and scale shapes don't match: {rshift.shape} != {scale.shape}")
        return rshift, scale

    @property
    @abstractmethod
    def _shift_scale_arr(self):
        pass

    def add_shift_scale(self, op):
        with self.using(op):
            shift_scale_arr = self._shift_scale_arr
        shift_scale_tensor = op.subgraph.create_tensor(
            f"{op.name}/shift_scale",
            TensorType.INT16,
            shift_scale_arr.shape,
            buffer=op.model.create_buffer(shift_scale_arr)
        )
        op.inputs.append(shift_scale_tensor)

    @abstractmethod
    def mutate_biases(self, op):
        pass

    def mutate_weights(self, op):
        with self.using(op):
            # rename weight tensor
            # NOTE: no weight layout rearrangement is done for this op
            self._weights.name = f"{op.name}/weights"

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        new_op = super().mutate(op)
        self.add_shift_scale(new_op)
        self.mutate_biases(new_op)
        self.mutate_weights(new_op)
        return new_op


# TODO: write (at least regression) tests for the mutator functions
class ReplaceDeepinAnyoutFullyConnectedPass(ReplaceXCOREWeightBiasOperatorPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.FULLY_CONNECTED

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._weights.shape[1] % 32 == 0

        return False

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

    @property
    def _shift_scale_arr(self):
        # calculate right shift/scale
        rshift, scale = self._shift_scale

        # reshape into appropriate array
        return np.hstack([rshift, scale]).reshape((2, -1))

    def mutate_output(self, op):
        with self.using(op):
            self._output.type = TensorType.INT16
            self._output.name = f"{op.name}/output"
            self._output.quantization = {
                'min': self._output.quantization['min'],
                'max': self._output.quantization['max'],
                'scale': [self._output.quantization['scale'][0] / 2**8],
                'zero_point': [int(self._output.quantization['zero_point'][0] * 2**8)],
                'details_type': "CustomQuantization",
                'quantized_dimension': 0
            }

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_fc_deepin_anyout)

    @abstractmethod
    def mutate(self, op):
        # NOTE: Overload this in subclasses, and call mutate_output appropriately
        return super().mutate(op)


class ReplaceDeepinAnyoutFullyConnectedOutputPass(ReplaceDeepinAnyoutFullyConnectedPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._output in op.subgraph.outputs

        return False

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        new_op = super().mutate(op)
        self.mutate_output(new_op)
        return new_op


# TODO: write (at least regression) tests for the mutator functions
class ReplaceDeepinAnyoutFullyConnectedIntermediatePass(ReplaceDeepinAnyoutFullyConnectedPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._output not in op.subgraph.outputs

        return False

    def add_requantize(self, op):
        # rename original output tensor
        with self.using(op):
            self._output.name = f"{op.name}/output_requant"
        # create intermediate tensor
        with self.using(op):
            intermediate = op.subgraph.create_tensor(
                f"{op.name}/intermediate", self._output.type, self._output.shape,
                quantization=self._output.quantization
            )
        # create new op, insert after original op, rewire inputs/outputs
        new_op = op.subgraph.create_operator(
            OperatorCode(XCOREOpCodes.XC_requantize_16_to_8),
            inputs=[intermediate], outputs=op.outputs)
        op.outputs = [intermediate]
        op.subgraph.insert_operator(op, new_op, after=True)

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        new_op = super().mutate(op)
        self.add_requantize(new_op)
        self.mutate_output(new_op)
        return new_op


# TODO: write (at least regression) tests for this class
class ReplaceDeepoutConv2DPass(ReplaceXCOREWeightBiasOperatorPass):
    @property
    def _strides(self):
        options = self._op.builtin_options
        return options['stride_h'], options['stride_w']

    @property
    def _dilation(self):
        options = self._op.builtin_options
        return options['dilation_h_factor'], options['dilation_w_factor']

    @property
    def _padding(self):
        return self._op.builtin_options['padding']

    def match(self, op):
        if super().match(op):
            with self.using(op):
                if self._dilation != (1, 1):
                    logging.warning(f"Found non-supported dilation: {self._dilation}")
                else:
                    return (self._strides == (1, 1)
                            and self._weights.shape[1] % 2 == 1  # kernel height is odd
                            and self._weights.shape[2] % 2 == 1  # kernel width is odd
                            and self._weights.shape[0] % 16 == 0)  # deepout

        return False

    def mutate_biases(self, op):
        with self.using(op):
            # calculate, reshape, and save a unified bias tensor
            bias = self._unified_bias
            acc_period = 16
            tmp_shape = (bias.shape[0] // acc_period, acc_period, -1)
            byte_list = list(bias.flatten().tostring())
            new_bias = np.uint8(byte_list).reshape(tmp_shape)
            new_bias = np.stack(  # splitting lower and upper 16 bits of each 32 bit value
                [new_bias[:, :, 2:], new_bias[:, :, :2]],
                axis=1
            )
            self._biases.buffer.data = new_bias

            # change bias tensor metadata and change quantization mode to alert users to unusual layout
            self._biases.type = TensorType.INT16
            self._biases.shape = list(new_bias.shape[:-1])
            self._biases.name = f"{op.name}/biases"
            self._biases.quantization['details_type'] = 'CustomQuantization'

    @abstractmethod
    def mutate_weights(self, op):
        def reorder_quant_params(arr, acc_period=16):
            arr = np.array(arr)
            arr = arr.reshape((arr.shape[0] // acc_period, acc_period))
            return np.flip(arr, axis=1).flatten().tolist()

        super().mutate_weights(op)
        with self.using(op):
            weight_quantization = self._weights.quantization
            for key in ['scale', 'zero_point']:
                weight_quantization[key] = reorder_quant_params(weight_quantization[key])

    @property
    def _shift_scale_arr(self):
        # calculate right shift/scale
        rshift, scale = self._shift_scale

        # reshape into appropriate array
        new_shape = (-1, 16)
        rshift = rshift.reshape(new_shape)  # pylint: disable=too-many-function-args
        scale = scale.reshape(new_shape)  # pylint: disable=too-many-function-args
        return np.stack([rshift, scale], axis=1)

    def mutate(self, op):
        new_op = super().mutate(op)
        with self.using(op):
            new_op.add_custom_options(
                padding=self._padding, stride_h=self._strides[0], stride_w=self._strides[1]
            )
        return new_op


# TODO: write (at least regression) tests for the mutator functions
class ReplaceDeepinDeepoutConv2DPass(ReplaceDeepoutConv2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._weights.shape[3] % 32 == 0  # deepin

        return False

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_conv2d_deepin_deepout_relu)

    def mutate_weights(self, op):
        super().mutate_weights(op)
        acc_period, ve = 16, 32  # parameters of the XS vector unit

        # rearrange weight tensor
        with self.using(op):
            weights = self._weights.numpy
            new_shape = (weights.shape[0] // acc_period, acc_period,
                         weights.shape[1], weights.shape[2],
                         weights.shape[3] // ve, ve)
            weights = weights.reshape(new_shape)
            weights = np.transpose(
                np.flip(weights, axis=1),
                axes=(0, 2, 3, 4, 1, 5)
            )

            # save weight tensor and update shape
            self._weights.buffer.data = np.int8(weights)
            self._weights.shape = list(weights.shape)


# TODO: write tests (of subclasses?) to test input operator matching
class ReplaceDeepoutConv2DInputPass(ReplaceDeepoutConv2DPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._input in op.subgraph.inputs

        return False

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_conv2d_shallowin_deepout_relu)

    def mutate_input(self, op):
        # NOTE: when trying to generalize this pass to non-input operators,
        #       keep in mind that this mutation is what can affect other operators
        with self.using(op):
            self._input.name = f"{op.name}/input"
            self._input.shape[3] = 4  # new, zero-padded shape

    def mutate_weights(self, op):
        # rearrange and zero pad weight tensor
        with self.using(op):
            weights = self._weights.numpy
            weights = np.pad(
                weights,
                pad_width=[(0, 0),
                           (0, 0),
                           (0, 8 - weights.shape[2]),
                           (0, 4 - weights.shape[3])]
            )
            acc_period = 16
            new_shape = (weights.shape[0] // acc_period, acc_period, weights.shape[1], 8, 4)
            weights = np.int8(weights.reshape(new_shape))
            weights = np.transpose(np.flip(weights, axis=1), axes=(0, 2, 1, 3, 4))

            self._weights.shape = weights.shape
            self._weights.buffer.data = weights

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        with self.using(op):
            unpadded_shape = self._weights.shape
        new_op = super().mutate(op)
        self.mutate_input(new_op)
        new_op.add_custom_options(unpadded_shape=unpadded_shape)
        return new_op


# TODO: write (at least regression) tests for the mutator functions
class ReplaceShallowinDeepoutConv2DPass(ReplaceDeepoutConv2DInputPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._weights.shape[3] <= 4  # shallowin
                        and self._weights.shape[2] <= 8)  # max kernel width

        return False


class ReplaceSingleinDeepoutDepthwiseConv2DPass(ReplaceDeepoutConv2DInputPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.DEPTHWISE_CONV_2D

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._weights.shape[3] == 1  # depthwise only matched with single input channel
                        and self._weights.shape[2] <= 8)  # max kernel width

        return False

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        with self.using(op):
            # NOTE: weight tensor channel ordering is:
            # kOHWI, // TFLite conv weights
            # kHWIO, // TensorFlow conv weights
            # k1HWO, // TFLite DepthwiseConv weights
            # kHWIM, // TensorFlow DepthwiseConv weights
            # Therefore, this permutation results in kOHW1 which is the same as
            #    TFLite conv weight order for a single input channel
            # NOTE: this happens before the standard weight mutation on purpose
            new_weights = np.transpose(self._weights.numpy, axes=(3, 1, 2, 0))
            self._weights.shape = new_weights.shape
            self._weights.buffer.data = new_weights
        return super().mutate(op)


class ReplaceDeepPooling2DPass(ReplaceQuantizedOperatorPass):
    @property
    def _strides(self):
        options = self._op.builtin_options
        return options['stride_h'], options['stride_w']

    @property
    def _pool_size(self):
        options = self._op.builtin_options
        return options['filter_height'], options['filter_width']

    @property
    def _padding(self):
        return self._op.builtin_options['padding']

    @property
    def _fused_activation(self):
        return self._op.builtin_options['fused_activation_function']

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._input.quantization == self._output.quantization
                        and self._padding == 'VALID'
                        and self._strides == (2, 2)
                        and self._pool_size == (2, 2)
                        and self._fused_activation == 'NONE'
                        and self._input.shape[1] % 2 == 0
                        and self._input.shape[2] % 2 == 0
                        and self._input.shape[3] % 32 == 0)

        return False


class ReplaceDeepMaxPool2DPass(ReplaceDeepPooling2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.MAX_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_maxpool2d_deep)


class ReplaceDeepAveragePool2DPass(ReplaceDeepPooling2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.AVERAGE_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_avgpool2d_deep)


class RemoveUnusedBuffersPass(ModelTransformationPass):
    def __init__(self, priority=PassPriority.CLEANUP):
        super().__init__(priority)

    def run(self, model):
        model.buffers = list(
            set(t.buffer for subgraph in model.subgraphs for t in subgraph.tensors)
            | set(m.buffer for m in model.metadata)
        )
