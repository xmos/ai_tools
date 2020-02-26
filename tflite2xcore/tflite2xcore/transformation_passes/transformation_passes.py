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
    OutputTensorMatchingPass,
    TensorMatchingPass
)
from tflite2xcore.operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from tflite2xcore.xcore_model import TensorType
from tflite2xcore.parallelization import DIDOConv2DPlanner

VE, ACC_PERIOD, WORD_SIZE = 32, 16, 4


class RemoveQuantizerFloatInputPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.PREP):
        super().__init__(priority)

    def match(self, op):
        if super().match(op) and op.operator_code.code == BuiltinOpCodes.QUANTIZE:
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
        if super().match(op) and op.operator_code.code == BuiltinOpCodes.DEQUANTIZE:
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
        return (super().match(input_tensor) and input_tensor.type == TensorType.INT8)

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
        return (super().match(input_tensor) and input_tensor.type == TensorType.INT8)

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
        return (super().match(op)
                and op.operator_code.code == BuiltinOpCodes.SOFTMAX
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
        return (super().match(tensor)
                and len(tensor.subgraph.outputs) == 1
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
            f"{op.name}/axis", TensorType.INT32, shape=[],
            consumers=[op])
        op.inputs.append(dim_tensor)
        dim_tensor.buffer.data = np.int32([1])


class QuantizedOperatorMatchingPass(OperatorMatchingPass):
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

    def match(self, op):
        if super().match(op) and op.operator_code.code == self.matching_opcode:
            with self.using(op):
                return (self._input.type == self._matching_input_type
                        and self._output.type == self._matching_output_type)


# TODO: write (at least regression) tests for this class
class ReplaceQuantizedOperatorPass(QuantizedOperatorMatchingPass):
    @property
    @abstractmethod
    def new_opcode(self):
        raise NotImplementedError()

    def mutate(self, op):
        new_op = op.subgraph.create_operator(
            self.new_opcode, inputs=op.inputs, outputs=op.outputs)
        new_op.subgraph.replace_operator(op, new_op)
        return new_op


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

    def __pad_to_acc_period(self, arr):
        pad = ACC_PERIOD - 1 - (arr.shape[0] - 1) % ACC_PERIOD
        return np.pad(arr, pad_width=[(0, pad)])

    @property
    def _bias_arr(self):
        # calculate bias values with the effect of quantization changes
        bias = self._unified_bias

        # zero pad and reshape
        bias = self.__pad_to_acc_period(bias)

        # splitting lower and upper 16 bits of each 32 bit value
        tmp_shape = (bias.shape[0] // ACC_PERIOD, ACC_PERIOD, -1)
        new_bias = np.frombuffer(bias.flatten().tostring(), dtype=np.int16).reshape(tmp_shape)
        return np.stack([new_bias[:, :, 1], new_bias[:, :, 0]], axis=1)

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
    def _shift_scale_arr(self):
        # calculate right shift/scale
        rshift, scale = self._shift_scale

        # zero pad and reshape into appropriate array
        new_shape = (-1, ACC_PERIOD)
        rshift = self.__pad_to_acc_period(rshift).reshape(new_shape)
        scale = self.__pad_to_acc_period(scale).reshape(new_shape)

        # split left and right shift into pre and post scaling shifts
        shift_pre = rshift if True else np.maximum(rshift, 0)  # TODO: resolve this when left shift issue is solved in conv2d kernels
        shift_post = 14 * np.ones(rshift.shape, dtype=rshift.dtype) + np.minimum(rshift, 0)
        return np.stack([shift_pre, scale, shift_post], axis=1)

    def mutate_biases(self, op):
        # NOTE: by default no bias layout rearrangement is done for this op
        with self.using(op):
            self._weights.name = f"{op.name}/biases"

    def mutate_weights(self, op):
        # NOTE: by default no weight layout rearrangement is done for this op
        with self.using(op):
            self._weights.name = f"{op.name}/weights"


# TODO: write (at least regression) tests for the mutator functions
class ReplaceFullyConnectedPass(ReplaceXCOREWeightBiasOperatorPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.FULLY_CONNECTED

    def mutate_weights(self, op):
        super().mutate_weights(op)
        with self.using(op):
            # zero_padding weight tensor
            col_pad = WORD_SIZE - 1 - (self._weights.shape[1] - 1) % WORD_SIZE
            arr = np.pad(
                self._weights.numpy.astype(np.int8),
                pad_width=[(0, 0),
                           (0, col_pad)]
            )

            # save weight tensor and update shape
            self._weights.buffer.data = arr
            self._weights.shape = arr.shape

            # remove quantization info to save space
            self._weights.quantization = None

    def mutate_biases(self, op):
        super().mutate_biases(op)
        with self.using(op):
            # calculate and save the bias/shift/scale tensor
            bss = np.concatenate([self._bias_arr, self._shift_scale_arr], axis=1)
            self._biases.buffer.data = bss
            self._biases.shape = bss.shape
            self._biases.type = TensorType.INT16

            # rename bias tensor and remove quantization info to save space
            self._biases.name = f"{op.name}/bias_shift_scale"
            self._biases.quantization = None

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
        # NOTE: the order of these mutations is strict
        new_op = super().mutate(op)
        self.mutate_biases(new_op)
        self.mutate_weights(new_op)
        return new_op


class ReplaceFullyConnectedOutputPass(ReplaceFullyConnectedPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._output in op.subgraph.outputs

        return False

    def mutate(self, op):
        new_op = super().mutate(op)
        self.mutate_output(new_op)
        return new_op


# TODO: write (at least regression) tests for the mutator functions
class ReplaceFullyConnectedIntermediatePass(ReplaceFullyConnectedPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._output not in op.subgraph.outputs

        return False

    def add_requantize(self, op):
        with self.using(op):
            # rename original output tensor
            self._output.name = f"{op.name}/output_requant"
            # create intermediate tensor
            intermediate = op.subgraph.create_tensor(
                f"{op.name}/intermediate", self._output.type, self._output.shape,
                quantization=self._output.quantization,
                producers=[op]
            )
        # rewire outputs of original FC op
        for output_tensor in op.outputs:
            output_tensor.producers.remove(op)
        # create new op, insert after original op, rewire inputs/outputs
        new_op = op.subgraph.create_operator(
            OperatorCode(XCOREOpCodes.XC_requantize_16_to_8),
            inputs=[intermediate], outputs=op.outputs)
        op.outputs = [intermediate]
        # move operator to correct location
        # TODO: remove this when we have execution planning
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
                            and self._weights.shape[0] % ACC_PERIOD == 0)  # deepout

        return False

    def mutate_biases(self, op):
        super().mutate_biases(op)
        with self.using(op):
            # calculate new bias tensor and save to buffer
            new_bias = self._bias_arr
            self._biases.buffer.data = new_bias

            # change bias tensor metadata
            self._biases.type = TensorType.INT16
            self._biases.shape = new_bias.shape

            # remove quantization info to save space
            self._biases.quantization = None

    def add_shift_scale(self, op):
        with self.using(op):
            shift_scale_arr = self._shift_scale_arr

        # TODO: remove this when left shift issue is solved in conv2d kernels
        shift_scale_arr = shift_scale_arr[:, :2, :]
        for s in shift_scale_arr[:, 0, :].flatten():
            if s < 0:
                raise ValueError("Negative right shift encountered.")

        shift_scale_tensor = op.subgraph.create_tensor(
            f"{op.name}/shift_scale", TensorType.INT16, shift_scale_arr.shape,
            buffer=op.model.create_buffer(shift_scale_arr),
            consumers=[op]
        )
        op.inputs.append(shift_scale_tensor)

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        new_op = super().mutate(op)
        self.add_shift_scale(new_op)
        self.mutate_biases(new_op)
        self.mutate_weights(new_op)

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
                return self._weights.shape[3] % VE == 0  # deepin

        return False

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_conv2d_deepin_deepout_relu)

    def mutate_weights(self, op):
        super().mutate_weights(op)
        with self.using(op):
            # rearrange weight tensor
            weights = self._weights.numpy.astype(np.int8)
            weights = weights.reshape((
                weights.shape[0] // ACC_PERIOD,
                ACC_PERIOD,
                weights.shape[1],
                weights.shape[2],
                weights.shape[3] // VE,
                VE
            ))
            weights = np.transpose(
                np.flip(weights, axis=1),
                axes=(0, 2, 3, 4, 1, 5)
            )

            # save weight tensor and update shape
            self._weights.buffer.data = weights
            self._weights.shape = weights.shape

            # remove quantization info to save space
            self._weights.quantization = None


# TODO: write tests (of subclasses?) to test input operator matching
class ReplaceDeepoutConv2DInputPass(ReplaceDeepoutConv2DPass):
    MAX_INPUT_CHANNELS = WORD_SIZE
    MAX_KERNEL_WIDTH = VE // MAX_INPUT_CHANNELS

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
            self._input.shape[3] = self.MAX_INPUT_CHANNELS  # new, zero-padded shape

    def mutate_weights(self, op):
        super().mutate_weights(op)
        with self.using(op):
            # rearrange and zero pad weight tensor
            weights = self._weights.numpy.astype(np.int8)
            weights = np.pad(
                weights,
                pad_width=[(0, 0),
                           (0, 0),
                           (0, self.MAX_KERNEL_WIDTH - weights.shape[2]),
                           (0, self.MAX_INPUT_CHANNELS - weights.shape[3])]
            )
            weights = weights.reshape((
                weights.shape[0] // ACC_PERIOD,
                ACC_PERIOD,
                weights.shape[1],
                self.MAX_KERNEL_WIDTH,
                self.MAX_INPUT_CHANNELS
            ))
            weights = np.transpose(
                np.flip(weights, axis=1),
                axes=(0, 2, 1, 3, 4)
            )

            # save weight tensor and update shape
            self._weights.shape = weights.shape
            self._weights.buffer.data = weights

            # remove quantization info to save space
            self._weights.quantization = None

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
                return (self._weights.shape[3] <= self.MAX_INPUT_CHANNELS
                        and self._weights.shape[2] <= self.MAX_KERNEL_WIDTH)

        return False


class ReplaceSingleinDeepoutDepthwiseConv2DPass(ReplaceDeepoutConv2DInputPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.DEPTHWISE_CONV_2D

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._weights.shape[3] == 1  # depthwise only matched with single input channel
                        and self._weights.shape[2] <= self.MAX_KERNEL_WIDTH)  # max kernel width

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


class ReplacePool2DPass(ReplaceQuantizedOperatorPass):
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
                        and self._fused_activation == 'NONE'
                        and self._input.shape[3] % 4 == 0)

        return False

    def mutate(self, op):
        new_op = super().mutate(op)

        with self.using(op):
            new_op.add_custom_options(
                stride_h=self._strides[0], stride_w=self._strides[1],
                pool_h=self._pool_size[0], pool_w=self._pool_size[1]
            )


class ReplacePool2D2x2Pass(ReplacePool2DPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._strides == (2, 2)
                        and self._pool_size == (2, 2)
                        and self._input.shape[1] % 2 == 0
                        and self._input.shape[2] % 2 == 0)

        return False


class ReplaceMaxPool2DPass(ReplacePool2DPass):
    def __init__(self, priority=PassPriority.MEDIUM, *, safe_mode=False):
        super().__init__(priority)
        self.safe_mode = safe_mode
        if self.safe_mode:
            self.superseding_passes.append(ReplaceMaxPool2D2x2Pass())

    @property
    def matching_opcode(self):
        return BuiltinOpCodes.MAX_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_maxpool2d)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._padding == 'VALID'

        return False


class ReplaceMaxPool2D2x2Pass(ReplacePool2D2x2Pass, ReplacePool2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.MAX_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_maxpool2d)


class ReplaceAveragePool2DPass(ReplacePool2DPass):
    def __init__(self, priority=PassPriority.MEDIUM, *, safe_mode=False):
        super().__init__(priority)
        self.safe_mode = safe_mode
        if self.safe_mode:
            self.superseding_passes.append(ReplaceAveragePool2D2x2Pass())

    @property
    def matching_opcode(self):
        return BuiltinOpCodes.AVERAGE_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_avgpool2d)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._padding == 'VALID'

        return False


class ReplaceAveragePool2D2x2Pass(ReplacePool2D2x2Pass, ReplacePool2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.AVERAGE_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_avgpool2d)


class ReplaceGlobalAveragePool2DPass(ReplaceQuantizedOperatorPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.MEAN

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_avgpool2d_global)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                reduction_dims = self._op.inputs[1].numpy
                return (len(reduction_dims) == 2
                        and np.all(reduction_dims == [1, 2])
                        and self._input.shape[3] % WORD_SIZE == 0)

        return False

    @property
    def _bias_scale_shift(self):
        num_pixels = self._input.shape[1] * self._input.shape[2]
        rescaling = self._input.quantization['scale'][0] / self._output.quantization['scale'][0]
        multiplier = rescaling / num_pixels

        scale = np.round(multiplier * 2 ** (7 - np.ceil(np.log2(multiplier))))
        shift = np.round(np.log2(scale / multiplier))
        bias = np.round(
            scale * (self._output.quantization['zero_point'][0] / multiplier
                     - self._input.quantization['zero_point'][0] * num_pixels)
        )

        if shift > 24:
            raise ValueError("Global Average Pool shift is greater than 24.")

        return bias.astype(np.int32), scale.astype(np.int8), shift.astype(np.int16)

    def mutate(self, op):
        new_op = super().mutate(op)
        subgraph = new_op.subgraph

        with self.using(new_op):
            # replace reduction_indices tensor with bias_scale_shift
            old_tensor = new_op.inputs[1]
            new_op.inputs[1] = subgraph.create_tensor(
                f"{new_op.name}/bias_scale_shift", TensorType.INT8, shape=[7],
                consumers=[new_op])
            new_op.inputs[1].buffer.data = np.frombuffer(
                b''.join(p.tostring() for p in self._bias_scale_shift),
                dtype=np.int8
            )
            subgraph.remove_tensor(old_tensor)

        return new_op


class RemoveUnusedBuffersPass(ModelTransformationPass):
    def __init__(self, priority=PassPriority.CLEANUP):
        super().__init__(priority)

    def run(self, model):
        model.buffers = [b for b in model.buffers if b.owners]


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


class ParallelizeDIDOPass(QuantizedOperatorMatchingPass):
    def __init__(self, priority=PassPriority.PAR, *, num_threads=None, forced=False):
        super().__init__(priority)
        self.num_threads = num_threads or 1
        assert isinstance(self.num_threads, int)
        assert self.num_threads > 0
        self.forced = forced

    def run(self, *args, **kwargs):
        if self.num_threads == 1:
            logging.debug(f"Skipping {type(self).__name__} with num_threads={self.num_threads}")
            return None
        else:
            return super().run(*args, **kwargs)

    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_conv2d_deepin_deepout_relu

    def match(self, op):
        if super().match(op):
            return 'par_plan' not in op.custom_options

    def mutate(self, op):
        with self.using(op):
            _, height, width, _ = self._output.shape
        planner = DIDOConv2DPlanner(
            height, width, num_threads=self.num_threads, forced=self.forced)
        plan = planner.find_optimal_plan()
        op.add_custom_options(par_plan=[list(block) for block in plan.layout])
