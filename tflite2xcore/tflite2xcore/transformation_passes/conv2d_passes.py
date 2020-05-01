# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np

from copy import deepcopy

from tflite2xcore.xcore_schema import (
    TensorType,
    BuiltinOpCodes,
    BuiltinOptions,
    OperatorCode,
    XCOREOpCodes,
)
from tflite2xcore.parallelization import DIDOConv2DPlanner, GenericConv2DPlanner
from tflite2xcore.utils import WORD_SIZE
from .transformation_passes import (
    ReplaceWeightBiasOperatorPass,
    QuantizedOperatorMatchingPass,
    LegalizeWeightBiasPass,
    LegalizeXCWeightBiasPass,
    OperatorMatchingPass,
)
from tflite2xcore.xlogging import log_method_output


class CanonicalizeSingleinDepthwiseConv2DPass(ReplaceWeightBiasOperatorPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.DEPTHWISE_CONV_2D

    @property
    def new_opcode(self):
        return OperatorCode(BuiltinOpCodes.CONV_2D, version=3)

    @property
    def _depth_multiplier(self):
        return self._op.builtin_options["depth_multiplier"]

    def match(self, op):
        with self.using(op):
            # TODO: update this when conv2d output channel word alignment is done
            return (
                super().match(op)
                and self._input.shape[3] == 1
                and self._output.shape[3] % WORD_SIZE == 0  # Cout divisible by 4
            )

    def mutate(self, op):
        with self.using(op):
            builtin_options = deepcopy(self._op.builtin_options)
            depth_multiplier = builtin_options.pop("depth_multiplier")
            assert depth_multiplier == self._weights.shape[3]

        # create new op and update builtin options
        new_op = super().mutate(op)
        new_op.builtin_options_type = BuiltinOptions.Conv2DOptions
        new_op.builtin_options = builtin_options

        return new_op


class LegalizeSingleinConv2DPass(LegalizeWeightBiasPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    def mutate_biases(self, op):
        # NOTE: nothing to be done on the biases
        pass

    def mutate_weights(self, op):
        with self.using(op):
            self._replace_weights(
                np.transpose(self._weights.numpy.astype(np.int8), [3, 1, 2, 0])
            )
            self._log_weights()


class ReplaceConv2DPass(ReplaceWeightBiasOperatorPass):
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


class LegalizeXCConvPass(LegalizeXCWeightBiasPass):
    @property
    def _MAX_POST_SHIFT(self):
        return 32 - 8 - 2  # this is because the output is 8 bit

    @property
    def _new_weight_shape(self):
        # by default, no reshaping is done
        return self._weights.shape

    def mutate_weights(self, op):
        with self.using(op):
            self._replace_weights(
                self._weights.numpy.astype(np.int8).reshape(self._new_weight_shape)
            )
            self._log_weights()


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


class LegalizeXC1x1ConvPass(LegalizeXCConvPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_conv2d_1x1

    @log_method_output()
    def _zero_point_bias(self):
        return np.sum(self._weights.numpy * self._input_zero_point, axis=3).squeeze()

    @property
    def _new_weight_shape(self):
        # NOTE: The reshape is not strictly necessary since the first dimension of
        #       the kernel should be 1 in TFLite
        old_shape = self._weights.shape
        return [old_shape[0], old_shape[3]]


class ReplacePaddedConv2DPass(ReplaceConv2DPass):
    def _pad(self):
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
        new_op = super().mutate(op)

        with self.using(op):
            new_op.add_custom_options(stride=self._strides)
        with self.using(new_op):
            new_op.add_custom_options(pad=self._pad())
        return new_op


class ReplaceDepthwiseConv2dPass(ReplacePaddedConv2DPass):
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


class LegalizeXCDepthwiseConvPass(LegalizeXCConvPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_conv2d_depthwise

    @log_method_output()
    def _zero_point_bias(self):
        # NOTE: first dimension of the kernel is always 1 in depthwise conv2d
        return np.sum(
            self._weights.numpy * self._input_zero_point, axis=(1, 2)
        ).squeeze()

    @property
    def _new_weight_shape(self):
        # NOTE: The reshape is not strictly necessary since the first dimension of
        #       the kernel should be 1 in TFLite
        return self._weights.shape[1:]


class ReplaceDeepConv2dPass(ReplacePaddedConv2DPass):
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
                    and self._weights.shape[3] % WORD_SIZE == 0  # Cin divisible by 4
                )

        return False


class LegalizeXCDeepConvPass(LegalizeXCConvPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_conv2d_deep

    @log_method_output()
    def _zero_point_bias(self):
        return np.sum(self._weights.numpy * self._input_zero_point, axis=(1, 2, 3))


class ReplaceShallowinConv2dPass(ReplacePaddedConv2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_conv2d_shallowin)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (
                    self._weights.shape[0] % WORD_SIZE == 0  # Cout divisible by 4
                    and self._weights.shape[3] % WORD_SIZE == 0  # Cin divisible by 4
                    and np.prod(self._weights.shape[2:]) <= 32  # K_w * Cin <= 32
                )

        return False

    def mutate(self, op):
        new_op = super().mutate(op)

        with self.using(new_op):
            new_op.add_custom_options(Kw=int(self._weights.shape[2]))
        return new_op


class LegalizeXCShallowinConvPass(LegalizeXCConvPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_conv2d_shallowin

    @log_method_output()
    def _zero_point_bias(self):
        return np.sum(self._weights.numpy * self._input_zero_point, axis=(1, 2, 3))

    def mutate_weights(self, op):
        with self.using(op):
            Kw_pad = int(32 / self._weights.shape[3] - self._weights.shape[2])
            unpadded_weights = self._weights.numpy.astype(np.int8).reshape(
                self._new_weight_shape
            )
            self._replace_weights(
                np.pad(
                    unpadded_weights, pad_width=[(0, 0), (0, 0), (0, Kw_pad), (0, 0)],
                )
            )
            self._log_weights()


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
