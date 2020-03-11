# Copyright (c) 2020, XMOS Ltd, All rights reserved
import logging
import numpy as np
from tflite2xcore.operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from tflite2xcore.xcore_model import TensorType
from tflite2xcore.graph_transformer import PassPriority
from tflite2xcore.utils import VE, ACC_PERIOD, WORD_SIZE
from .transformation_passes import ReplaceXCOREWeightBiasOperatorPass


class ReplaceConv2DPass(ReplaceXCOREWeightBiasOperatorPass):
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
                    return True

        return False

    @property
    def _MAX_POST_SHIFT(self):
        return 32 - 8 - 2  # this is because the output is 8 bit


class Replace1x1Conv2dPass(ReplaceConv2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_conv2d_1x1)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._strides == (1, 1)
                        and self._weights.shape[0] % WORD_SIZE == 0  # Cout divisible by 4
                        and self._weights.shape[1] == 1
                        and self._weights.shape[2] == 1
                        and self._weights.shape[3] % WORD_SIZE == 0)  # Cin divisible by 4

        return False

    def mutate_biases(self, op):
        # TODO: this is the same as in ReplaceFullyConnectedPass, refactor
        super().mutate_biases(op)
        with self.using(op):
            # calculate and save the bias/shift/scale tensor
            bss = self._bss_arr
            self._biases.buffer.data = bss
            self._biases.shape = bss.shape
            self._biases.type = TensorType.INT16

            # rename bias tensor and remove quantization info to save space
            self._biases.name = f"{op.name}/bias_shift_scale"
            self._biases.quantization = None

    def mutate_weights(self, op):
        super().mutate_weights(op)
        with self.using(op):
            # NOTE: This is not strictly necessary since height == width == 1
            old_shape = self._weights.shape
            self._weights.shape = [old_shape[0], old_shape[3]]

            # remove quantization info to save space
            self._weights.quantization = None

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        new_op = super().mutate(op)
        self.mutate_biases(new_op)
        self.mutate_weights(new_op)

        with self.using(op):
            new_op.add_custom_options(
                stride_h=self._strides[0], stride_w=self._strides[1]
            )
        return new_op


# TODO: write (at least regression) tests for this class
class ReplaceDeepoutConv2DPass(ReplaceConv2DPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
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
    def __init__(self, priority=PassPriority.MEDIUM, *, safe_mode=False):
        super().__init__(priority)
        self.safe_mode = safe_mode
        if self.safe_mode:
            self.superseding_passes.append(Replace1x1Conv2dPass())

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
            self._input.shape = [*self._input.shape[:3], self.MAX_INPUT_CHANNELS]  # new, zero-padded shape

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
