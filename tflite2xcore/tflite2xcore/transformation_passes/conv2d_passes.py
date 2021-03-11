# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import numpy as np
from copy import deepcopy

from tflite2xcore.xcore_schema import (
    BuiltinOpCodes,
    OperatorCode,
    XCOREOpCodes,
    FullyConnectedOptionsWeightsFormat,
)
from tflite2xcore.utils import WORD_SIZE_BYTES

from .transformation_passes import (
    ReplaceQuantizedWeightBiasOperatorPass,
    ReplaceXCWeightBiasOperatorPass,
    LegalizeWeightBiasPass,
    LegalizeXCWeightBiasPass,
)


class CanonicalizeSinglePixelConv2DPass(ReplaceQuantizedWeightBiasOperatorPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    @property
    def new_opcode(self):
        return OperatorCode(BuiltinOpCodes.FULLY_CONNECTED, version=7)

    def match(self, op):
        with self.using(op):
            return (
                super().match(op)
                and self._weights.shape[1:3] == self._input.shape[1:3]
                and self._output.shape[1] == self._output.shape[2] == 1
            )

    def mutate(self, op):
        builtin_options = {
            "fused_activation_function": op.builtin_options[
                "fused_activation_function"
            ],
            "weights_format": FullyConnectedOptionsWeightsFormat.DEFAULT,
            "keep_num_dims": False,
            "asymmetric_quantize_inputs": False,
        }

        new_op = super().mutate(op)
        with self.using(new_op):
            old_weight_tensor = self._weights
            self._op.builtin_options = builtin_options

            new_weight_tensor = self._op.subgraph.create_tensor(
                f"{self._op.name}/weights",
                old_weight_tensor.type,
                shape=(
                    old_weight_tensor.shape[0],
                    np.prod(old_weight_tensor.shape[1:]),
                ),
                quantization=old_weight_tensor.quantization,
                consumers=[self._op],
                buffer=old_weight_tensor.buffer,
            )

            # rewire old and new kernel tensors
            old_weight_tensor.consumers.remove(self._op)
            self._op.inputs[1] = new_weight_tensor

        return new_op


class CanonicalizeSingleinDepthwiseConv2DPass(ReplaceXCWeightBiasOperatorPass):
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
                and self._output.shape[3] % WORD_SIZE_BYTES == 0  # Cout divisible by 4
            )

    def mutate(self, op):
        with self.using(op):
            builtin_options = deepcopy(self._op.builtin_options)
            depth_multiplier = builtin_options.pop("depth_multiplier")
            assert depth_multiplier == self._weights.shape[3]

        # create new op and update builtin options
        new_op = super().mutate(op)
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
            self._replace_weights(np.transpose(self._weights.as_array(), [3, 1, 2, 0]))


class ReplaceConv2DPass(ReplaceXCWeightBiasOperatorPass):
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
    def _new_weight_shape(self):
        # by default, no reshaping is done
        return self._weights.shape

    def mutate_weights(self, op):
        with self.using(op):
            self._replace_weights(
                self._weights.as_array().reshape(self._new_weight_shape)
            )


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
                    and self._weights.shape[0] % WORD_SIZE_BYTES
                    == 0  # Cout divisible by 4
                    and self._weights.shape[1] == 1
                    and self._weights.shape[2] == 1
                    and self._weights.shape[3] % WORD_SIZE_BYTES
                    == 0  # Cin divisible by 4
                )

        return False


class LegalizeXC1x1ConvPass(LegalizeXCConvPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_conv2d_1x1

    def _zero_point_bias(self):
        return np.sum(
            self._weights.as_array(np.int64) * self._input_zero_point, axis=3
        ).squeeze()

    @property
    def _new_weight_shape(self):
        # NOTE: The reshape is not strictly necessary since the first dimension of
        #       the kernel should be 1 in TFLite
        old_shape = self._weights.shape
        return [old_shape[0], old_shape[3]]


class ReplacePaddedConv2DPass(ReplaceConv2DPass):
    def _pad(self):
        # pad: [top, left, zero_point]
        pad = tuple(
            # first arg of max is <= for valid padding
            max(int((o - 1) * s - i + k) // 2, 0)
            for o, s, i, k in zip(
                self._output.shape[1:3],
                self._op.custom_options["stride"],
                self._input.shape[1:3],
                self._weights.shape[1:3],
            )
        )
        return (*pad, self._input_zero_point)

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
                    return (
                        self._weights.shape[3] % WORD_SIZE_BYTES == 0
                    )  # Cin divisible by 4

        return False


class LegalizeXCDepthwiseConvPass(LegalizeXCConvPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_conv2d_depthwise

    def _zero_point_bias(self):
        # NOTE: first dimension of the kernel is always 1 in depthwise conv2d
        return np.sum(
            self._weights.as_array(np.int64) * self._input_zero_point, axis=(1, 2)
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
                    self._weights.shape[0] % WORD_SIZE_BYTES == 0  # Cout divisible by 4
                    and self._weights.shape[3] % WORD_SIZE_BYTES
                    == 0  # Cin divisible by 4
                )

        return False


class LegalizeXCDeepConvPass(LegalizeXCConvPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_conv2d_deep

    def _zero_point_bias(self):
        return np.sum(
            self._weights.as_array(np.int64) * self._input_zero_point, axis=(1, 2, 3)
        )


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
                    self._weights.shape[0] % WORD_SIZE_BYTES == 0  # Cout divisible by 4
                    and self._weights.shape[3] % WORD_SIZE_BYTES
                    == 0  # Cin divisible by 4
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

    def _zero_point_bias(self):
        return np.sum(
            self._weights.as_array(np.int64) * self._input_zero_point, axis=(1, 2, 3)
        )

    def mutate_weights(self, op):
        with self.using(op):
            Kw_pad = int(32 / self._weights.shape[3] - self._weights.shape[2])
            unpadded_weights = self._weights.as_array().reshape(self._new_weight_shape)
            self._replace_weights(
                np.pad(
                    unpadded_weights, pad_width=[(0, 0), (0, 0), (0, Kw_pad), (0, 0)],
                )
            )
