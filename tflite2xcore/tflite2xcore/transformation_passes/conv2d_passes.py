# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np
from tflite2xcore.operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from tflite2xcore.xcore_model import TensorType
from tflite2xcore.parallelization import DIDOConv2DPlanner, GenericConv2DPlanner
from tflite2xcore.utils import VE, ACC_PERIOD, WORD_SIZE
from .transformation_passes import (
    ReplaceXCOREWeightBiasOperatorPass,
    QuantizedOperatorMatchingPass,
    OperatorMatchingPass,
)
from tflite2xcore.xlogging import log_method_output


class ReplaceConv2DPass(ReplaceXCOREWeightBiasOperatorPass):
    @property
    def _strides(self):
        options = self._op.builtin_options
        return options["stride_h"], options["stride_w"]

    @property
    def _dilation(self):
        options = self._op.builtin_options
        return options["dilation_h_factor"], options["dilation_w_factor"]

    @property
    def _padding(self):
        return self._op.builtin_options["padding"]

    def match(self, op):
        if super().match(op):
            with self.using(op):
                if self._dilation != (1, 1):
                    self.logger.warning(
                        f"Found non-supported dilation: {self._dilation}"
                    )
                else:
                    return True

        return False

    @property
    def _MAX_POST_SHIFT(self):
        return 32 - 8 - 2  # this is because the output is 8 bit


class ReplaceDepthwiseConv2dPass(ReplaceConv2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.DEPTHWISE_CONV_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_conv2d_depthwise)

    @property
    def _depth_multiplier(self):
        return self._op.builtin_options["depth_multiplier"]

    def match(self, op):
        if super().match(op):
            with self.using(op):
                if self._depth_multiplier != 1:
                    self.logger.warning(
                        f"Found non-supported depthwise multiplier: {self._depth_multiplier}"
                    )
                else:
                    return self._weights.shape[3] % WORD_SIZE == 0  # Cin divisible by 4

        return False

    @log_method_output()
    def _zero_point_bias(self):
        # NOTE: first dimension of the kernel is always 1 in depthwise conv2d
        return np.sum(
            self._weights.numpy * self._input_zero_point, axis=(1, 2)
        ).squeeze()

    def mutate_biases(self, op):
        # TODO: this is the same as in ReplaceFullyConnectedPass, refactor
        # TODO: this is the same as in Replace1x1Conv2dPass, refactor
        # TODO: this is the same as in ReplaceDeepConv2dPass, refactor
        super().mutate_biases(op)
        with self.using(op):
            # calculate and save the bias/shift/scale tensor
            bss = self._bss_arr()
            self._biases.buffer.data = bss
            self._biases.shape = bss.shape
            self._biases.type = TensorType.INT16
            self._biases.name = f"{op.name}/bias_shift_scale"

    def mutate_weights(self, op):
        super().mutate_weights(op)
        with self.using(op):
            # NOTE: This is not strictly necessary since the first dimension of
            #       the kernel should be 1 in TFLite
            self._weights.shape = self._weights.shape[1:]
            self._log_weights()

    def _pad(self):
        # TODO: this is very similar to the one in ReplaceDeepConv2dPass, refactor
        # pad: [top, left, zero_point]
        pad = [
            max(int((o - 1) * s - i + k) // 2, 0)
            for o, s, i, k in zip(
                self._output.shape[1:3],
                self._op.custom_options["stride"],
                self._input.shape[1:3],
                self._weights.shape[0:2],
            )
        ]
        pad.append(self._input_zero_point)
        return pad

    def mutate(self, op):
        # TODO: this is the same as in Replace1x1Conv2dPass, refactor
        # NOTE: the order of these mutations is strict
        new_op = super().mutate(op)
        self.mutate_biases(new_op)
        self.mutate_weights(new_op)

        with self.using(op):
            new_op.add_custom_options(stride=self._strides)
        with self.using(new_op):
            new_op.add_custom_options(pad=self._pad())
        return new_op


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
                return (
                    self._strides == (1, 1)
                    and self._weights.shape[0] % WORD_SIZE == 0  # Cout divisible by 4
                    and self._weights.shape[1] == 1
                    and self._weights.shape[2] == 1
                    and self._weights.shape[3] % WORD_SIZE == 0  # Cin divisible by 4
                )

        return False

    @log_method_output()
    def _zero_point_bias(self):
        return np.sum(self._weights.numpy * self._input_zero_point, axis=3).squeeze()

    def mutate_biases(self, op):
        # TODO: this is the same as in ReplaceFullyConnectedPass, refactor
        # TODO: this is the same as in ReplaceDepthwiseConv2dPass, refactor
        # TODO: this is the same as in ReplaceDeepConv2dPass, refactor
        super().mutate_biases(op)
        with self.using(op):
            # calculate and save the bias/shift/scale tensor
            bss = self._bss_arr()
            self._biases.buffer.data = bss
            self._biases.shape = bss.shape
            self._biases.type = TensorType.INT16
            self._biases.name = f"{op.name}/bias_shift_scale"

    def mutate_weights(self, op):
        super().mutate_weights(op)
        with self.using(op):
            # NOTE: This is not strictly necessary since height == width == 1
            old_shape = self._weights.shape
            self._weights.shape = [old_shape[0], old_shape[3]]
            self._log_weights()

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        # TODO: this is the same as in ReplaceDepthwiseConv2dPass, refactor
        new_op = super().mutate(op)
        self.mutate_biases(new_op)
        self.mutate_weights(new_op)

        return new_op


class ReplaceDeepConv2dPass(ReplaceConv2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_conv2d_deep)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (
                    self._weights.shape[0] % WORD_SIZE == 0  # Cout divisible by 4
                    and self._weights.shape[3] % WORD_SIZE == 0
                )  # Cin divisible by 4

        return False

    # TODO: refactor this
    @log_method_output()
    def _zero_point_bias(self):
        return np.sum(self._weights.numpy * self._input_zero_point, axis=(1, 2, 3))

    def mutate_biases(self, op):
        # TODO: this is the same as in ReplaceFullyConnectedPass, refactor
        # TODO: this is the same as in ReplaceDepthwiseConv2dPass, refactor
        # TODO: this is the same as in Replace1x1Conv2dPass, refactor
        super().mutate_biases(op)
        with self.using(op):
            # calculate and save the bias/shift/scale tensor
            bss = self._bss_arr()
            self._biases.buffer.data = bss
            self._biases.shape = bss.shape
            self._biases.type = TensorType.INT16
            self._biases.name = f"{op.name}/bias_shift_scale"

    def _pad(self):
        # TODO: this is very similar to the one in ReplaceDepthwiseConv2dPass, refactor
        # pad: [top, left, zero_point]
        pad = [
            max(int((o - 1) * s - i + k) // 2, 0)
            for o, s, i, k in zip(
                self._output.shape[1:3],
                self._op.custom_options["stride"],
                self._input.shape[1:3],
                self._weights.shape[1:3],
            )
        ]
        pad.append(self._input_zero_point)
        return pad

    def mutate(self, op):
        # TODO: this is the same as in ReplaceDepthwiseConv2dPass, refactor
        # TODO: this is the same as in Replace1x1Conv2dPass, refactor
        # NOTE: the order of these mutations is strict
        new_op = super().mutate(op)
        self.mutate_biases(new_op)
        self.mutate_weights(new_op)

        with self.using(op):
            new_op.add_custom_options(stride=self._strides)
        with self.using(new_op):
            new_op.add_custom_options(pad=self._pad())
        return new_op


class ParallelizeXCConv2dPass(OperatorMatchingPass):
    def __init__(self, *args, num_threads=None, forced=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_threads = num_threads or 1
        assert isinstance(self.num_threads, int)
        assert self.num_threads > 0
        self.forced = forced

    MATCHING_OPCODES = (
        XCOREOpCodes.XC_conv2d_depthwise,
        XCOREOpCodes.XC_conv2d_deep,
        XCOREOpCodes.XC_conv2d_1x1,
    )

    def run(self, *args, **kwargs):
        if self.num_threads == 1:
            self.logger.debug(f"Skipping pass b/c num_threads={self.num_threads}")
            return 0
        else:
            return super().run(*args, **kwargs)

    def match(self, op):
        if super().match(op) and op.operator_code.code in self.MATCHING_OPCODES:
            return "par_plan" not in op.custom_options

    def mutate(self, op):
        _, height, width, _ = op.outputs[0].shape
        assert int(height) == height
        assert int(width) == width
        planner = GenericConv2DPlanner(
            int(height), int(width), num_threads=self.num_threads, forced=self.forced
        )
        plan = planner.find_optimal_plan()
        op.add_custom_options(par_plan=[list(block) for block in plan.layout])


# -----------------------------------------------------------------------------
#                             DEPRECATED FUNCTiONS
# -----------------------------------------------------------------------------


# TODO: Consider deprecating this when conv2d enhancements are done
class ReplaceDeepoutConv2DPass(ReplaceConv2DPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (
                    self._strides == (1, 1)
                    and self._weights.shape[1] % 2 == 1  # kernel height is odd
                    and self._weights.shape[2] % 2 == 1  # kernel width is odd
                    and self._output.shape[3] % ACC_PERIOD == 0
                )  # deepout

        return False

    @log_method_output()
    def _zero_point_bias(self):
        return np.sum(self._weights.numpy * self._input_zero_point, axis=(1, 2, 3))

    def mutate_biases(self, op):
        super().mutate_biases(op)
        with self.using(op):
            # calculate new bias tensor and save to buffer
            new_bias = self._bias_arr()
            self._biases.buffer.data = new_bias

            # change bias tensor metadata
            self._biases.type = TensorType.INT16
            self._biases.shape = new_bias.shape

            # remove quantization info to save space
            self._biases.quantization = None

    def add_shift_scale(self, op):
        with self.using(op):
            shift_scale_arr = self._shift_scale_arr()

        # TODO: remove this when left shift issue is solved in conv2d kernels
        shift_scale_arr = shift_scale_arr[:, :2, :]
        for s in shift_scale_arr[:, 0, :].flatten():
            if s < 0:
                raise ValueError("Negative right shift encountered.")

        shift_scale_tensor = op.subgraph.create_tensor(
            f"{op.name}/shift_scale",
            TensorType.INT16,
            shift_scale_arr.shape,
            buffer=op.model.create_buffer(shift_scale_arr),
            consumers=[op],
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
                padding=self._padding,
                stride_h=self._strides[0],
                stride_w=self._strides[1],  # TODO: change to 'stride' and 'pad'
            )
        return new_op


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
            self._input.shape = [
                *self._input.shape[:3],
                self.MAX_INPUT_CHANNELS,
            ]  # new, zero-padded shape

    def mutate_weights(self, op):
        super().mutate_weights(op)
        with self.using(op):
            # rearrange and zero pad weight tensor
            weights = self._weights.numpy.astype(np.int8)
            weights = np.pad(
                weights,
                pad_width=[
                    (0, 0),
                    (0, 0),
                    (0, self.MAX_KERNEL_WIDTH - weights.shape[2]),
                    (0, self.MAX_INPUT_CHANNELS - weights.shape[3]),
                ],
            )
            weights = weights.reshape(
                (
                    weights.shape[0] // ACC_PERIOD,
                    ACC_PERIOD,
                    weights.shape[1],
                    self.MAX_KERNEL_WIDTH,
                    self.MAX_INPUT_CHANNELS,
                )
            )
            weights = np.transpose(np.flip(weights, axis=1), axes=(0, 2, 1, 3, 4))
            self._log_weights()

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
                return (
                    self._weights.shape[3] <= self.MAX_INPUT_CHANNELS
                    and self._weights.shape[2] <= self.MAX_KERNEL_WIDTH
                )

        return False


class ReplaceSingleinDeepoutDepthwiseConv2DPass(ReplaceDeepoutConv2DInputPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.DEPTHWISE_CONV_2D

    def match(self, op):
        if super().match(op):
            with self.using(op):
                print(self._input.shape)
                return (
                    self._input.shape[3]
                    == 1  # depthwise only matched with single input channel
                    and self._weights.shape[2] <= self.MAX_KERNEL_WIDTH
                )  # max kernel width

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
            new_weights = np.transpose(
                self._weights.numpy.astype(np.int8), axes=(3, 1, 2, 0)
            )
            self._weights.shape = new_weights.shape
            self._weights.buffer.data = new_weights
        return super().mutate(op)


class ParallelizeDeepConv2dPass(QuantizedOperatorMatchingPass):
    def __init__(self, *args, num_threads=None, forced=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_threads = num_threads or 1
        assert isinstance(self.num_threads, int)
        assert self.num_threads > 0
        self.forced = forced

    def run(self, *args, **kwargs):
        if self.num_threads == 1:
            self.logger.debug(f"Skipping pass b/c num_threads={self.num_threads}")
            return 0
        else:
            return super().run(*args, **kwargs)

    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_conv2d_deep

    def match(self, op):
        if super().match(op):
            return "par_plan" not in op.custom_options

    def mutate(self, op):
        with self.using(op):
            _, height, width, _ = self._output.shape
        assert int(height) == height
        assert int(width) == width
        planner = DIDOConv2DPlanner(
            int(height), int(width), num_threads=self.num_threads, forced=self.forced
        )
        plan = planner.find_optimal_plan()
        op.add_custom_options(par_plan=[list(block) for block in plan.layout])
