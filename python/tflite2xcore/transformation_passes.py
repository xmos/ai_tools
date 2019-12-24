# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from contextlib import contextmanager
from .graph_transformer import PassPriority
from .graph_transformer import (
    ModelTransformationPass,
    OperatorMatchingPass,
    InputTensorMatchingPass,
    OutputTensorMatchingPass
)
from .operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from .xcore_model import TensorType


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
        subgraph.inputs.append(fin)
        subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.QUANTIZE), inputs=[fin], outputs=[qin])


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
        subgraph.outputs.append(fout)
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
            f"{tensor.name}_argmax", tensor.type, tensor.shape, isoutput=True)
        subgraph.outputs.remove(tensor)
        subgraph.outputs.append(tout)
        subgraph.create_operator(
            OperatorCode(XCOREOpCodes.XC_argmax_16), inputs=[tensor], outputs=[tout])


# TODO: write (at least regression) tests for this class
class ReplaceQuantizedWeightBiasOperatorPass(OperatorMatchingPass):
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
    def _weights(self):
        return self._op.inputs[1]

    @property
    def _biases(self):
        return self._op.inputs[2]

    @abstractmethod
    def _match_opcode(self, op):
        raise NotImplementedError()

    def match(self, op):
        if self._match_opcode(op):
            with self.using(op):
                return (self._weights.type == TensorType.INT8
                        and self._input.type == TensorType.INT8
                        and self._output.type == TensorType.INT8
                        and self._biases.type == TensorType.INT32)

    @property
    @abstractmethod
    def new_opcode(self):
        raise NotImplementedError()

    def mutate(self, op):
        new_op = op.subgraph.create_operator(
            self.new_opcode, inputs=op.inputs, outputs=op.outputs)
        op.subgraph.remove_operator(op)
        return new_op


# TODO: write (at least regression) tests for this class
class ReplaceXCOREWeightBiasOperatorPass(ReplaceQuantizedWeightBiasOperatorPass):
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

    @abstractmethod
    def mutate_biases(self, op):
        pass

    @abstractmethod
    def mutate_weights(self, op):
        pass

    @abstractmethod
    def mutate_output(self, op):
        pass

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        op = super().mutate(op)
        self.add_shift_scale(op)
        self.mutate_biases(op)
        self.mutate_weights(op)
        self.mutate_output(op)


# TODO: write (at least regression) tests for the mutator functions
class ReplaceDeepinShallowoutFullyConnectedOutputPass(ReplaceXCOREWeightBiasOperatorPass):
    def _match_opcode(self, op):
        return op.operator_code.code == BuiltinOpCodes.FULLY_CONNECTED

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._output in op.subgraph.outputs
                        and self._weights.shape[0] < 16
                        and self._weights.shape[1] % 32 == 0)

        return False

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_fc_deepin_shallowout_final)

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
                return (self._dilation == (1, 1)
                        and self._strides == (1, 1)
                        and self._padding == 'SAME'
                        and self._weights.shape[1] % 2 == 1  # kernel height is odd
                        and self._weights.shape[2] % 2 == 1  # kernel width is odd
                        and self._weights.shape[0] % 16 == 0)  # deepout

        return False

    # TODO: it would be better to do this without tensorflow
    @property
    def _pad_bias(self):
        _, K_h, K_w, C_in = self._weights.shape
        pad_b = pad_t = K_h//2
        pad_l = pad_r = K_w//2

        # create a template with the desired padding of zero_point values
        input_zero_point = int(self._input.quantization['zero_point'][0])
        pad_template = tf.pad(
            tf.zeros(shape=(1, K_h, K_w, C_in), dtype=tf.float32),
            paddings=[(0, 0), (pad_b, pad_t), (pad_l, pad_r), (0, 0)],
            constant_values=input_zero_point
        )

        # run a convolution with VALID padding
        # this results in a tensor identical in size to the kernel
        # NOTE: each element in this output captures the effect of the zero point
        #       shift of the input tensor that is not accounted for when applying
        #       zero padding in the kernel implementation
        filters = tf.transpose(
            tf.convert_to_tensor(self._weights.numpy, dtype=tf.float32),
            perm=(1, 2, 3, 0)
        )
        pad_effects = tf.nn.conv2d(
            input=pad_template, filters=filters, strides=1, padding='VALID')
        pad_bias = tf.dtypes.cast(
            pad_effects + self._unified_bias.reshape((1, 1, -1)),  # pylint: disable=too-many-function-args
            dtype=tf.int32
        )

        return pad_bias.numpy()

    def mutate_biases(self, op):
        with self.using(op):
            # calculate, reshape, and save a unified bias tensor with pad effects
            pad_bias = self._pad_bias
            tmp_shape = list(pad_bias.shape[1:])+[-1]
            byte_list = list(pad_bias.flatten().tostring())
            new_bias = np.uint8(byte_list).reshape(tmp_shape)
            new_bias = np.stack(  # splitting lower and upper 16 bits of each 32 bit value
                [new_bias[:, :, :, :2], new_bias[:, :, :, 2:]],
                axis=2
            )
            self._biases.buffer.data = new_bias

            # change bias tensor metadata and change quantization mode to alert users to unusual layout
            self._biases.type = TensorType.INT16
            self._biases.shape = list(new_bias.shape[:-1])
            self._biases.name = f"{op.name}/biases_padded"
            self._biases.quantization['details_type'] = 'CustomQuantization'

    @abstractmethod
    def mutate_weights(self, op):
        def reorder_quant_params(arr, acc_period=16):
            arr = np.array(arr)
            arr = arr.reshape((arr.shape[0] // acc_period, acc_period))
            return np.flip(arr, axis=1).flatten().tolist()

        with self.using(op):
            weight_quantization = self._weights.quantization
            for key in ['scale', 'zero_point']:
                weight_quantization[key] = reorder_quant_params(weight_quantization[key])

    def mutate_output(self, op):
        # this pass should not modify the output
        pass


# TODO: write (at least regression) tests for the mutator functions
class ReplaceDeepinDeepoutConv2DPass(ReplaceDeepoutConv2DPass):
    def _match_opcode(self, op):
        return op.operator_code.code == BuiltinOpCodes.CONV_2D

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


# TODO: write (at least regression) tests for the mutator functions
class ReplaceShallowinDeepoutConv2DPass(ReplaceDeepoutConv2DPass):
    def _match_opcode(self, op):
        return op.operator_code.code == BuiltinOpCodes.CONV_2D

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._weights.shape[3] <= 4  # shallowin
                        and self._weights.shape[2] <= 8)  # max kernel width

        return False

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_conv2d_shallowin_deepout_relu)

    def mutate_weights(self, op):
        raise NotImplementedError()


class ReplaceSingleinDeepoutDepthwiseConv2DPass(ReplaceDeepoutConv2DPass):
    def _match_opcode(self, op):
        return op.operator_code.code == BuiltinOpCodes.DEPTHWISE_CONV_2D

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._weights.shape[3] == 1  # depthwise only matched with single input channel
                        and self._weights.shape[2] <= 8)  # max kernel width

        return False

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_conv2d_shallowin_deepout_relu)

    def mutate_weights(self, op):
        raise NotImplementedError()


class RemoveUnusedBuffersPass(ModelTransformationPass):
    def __init__(self, priority=PassPriority.CLEANUP):
        super().__init__(priority)

    def run(self, model):
        model.buffers = list(
            set(t.buffer for subgraph in model.subgraphs for t in subgraph.tensors)
            | set(m.buffer for m in model.metadata)
        )
