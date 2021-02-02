# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import numpy as np

from tflite2xcore.xcore_schema import (
    ActivationFunctionType,
    Padding,
    TensorType,
    BuiltinOpCodes,
    OperatorCode,
    XCOREOpCodes,
)
from tflite2xcore.utils import WORD_SIZE_BYTES

from .transformation_passes import ReplaceQuantizedOperatorPass


class ReplacePool2DPass(ReplaceQuantizedOperatorPass):
    @property
    def _strides(self):
        options = self._op.builtin_options
        return options["stride_h"], options["stride_w"]

    @property
    def _pool_size(self):
        options = self._op.builtin_options
        return options["filter_height"], options["filter_width"]

    @property
    def _padding(self):
        return self._op.builtin_options["padding"]

    @property
    def _fused_activation(self):
        return self._op.builtin_options["fused_activation_function"]

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (
                    self._input.quantization == self._output.quantization
                    and self._fused_activation is ActivationFunctionType.NONE
                    and self._input.shape[3] % 4 == 0
                )

        return False

    def mutate(self, op):
        new_op = super().mutate(op)

        with self.using(op):
            new_op.add_custom_options(stride=self._strides, pool=self._pool_size)


class ReplacePool2D2x2Pass(ReplacePool2DPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (
                    self._strides == (2, 2)
                    and self._pool_size == (2, 2)
                    and self._input.shape[1] % 2 == 0
                    and self._input.shape[2] % 2 == 0
                )

        return False


class ReplaceMaxPool2DPass(ReplacePool2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.MAX_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_maxpool2d)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._padding is Padding.VALID

        return False


class ReplaceMaxPool2D2x2Pass(ReplacePool2D2x2Pass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.MAX_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_maxpool2d)


class ReplaceAveragePool2DPass(ReplacePool2DPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.AVERAGE_POOL_2D

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_avgpool2d)

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._padding is Padding.VALID

        return False


class ReplaceAveragePool2D2x2Pass(ReplacePool2D2x2Pass):
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
                axis = self._op.inputs[1].as_array().flatten().tolist()
                if axis == [1, 2] or axis == [2, 1]:
                    return self._input.shape[3] % WORD_SIZE_BYTES == 0
                else:
                    self.logger.warning("Axis is not either [1, 2] or [2, 1]")

        return False

    @property
    def _bias_scale_shift(self):
        num_pixels = self._input.shape[1] * self._input.shape[2]
        rescaling = (
            self._input.quantization["scale"][0] / self._output.quantization["scale"][0]
        )
        multiplier = rescaling / num_pixels

        scale = np.round(multiplier * 2 ** (7 - np.ceil(np.log2(multiplier))))
        if scale == 128.0:
            scale /= 2
        shift = np.round(np.log2(scale / multiplier))
        bias = np.round(
            (
                self._output.quantization["zero_point"][0]
                - self._input.quantization["zero_point"][0] * rescaling
                + 0.5  # needed because the tflite ref adds 0.5 to the bias
            )
            * 2 ** shift
        )

        if shift > 24 or shift < 0:
            raise ValueError(
                f"Global Average Pool shift must be between 0 and 24, got {shift}."
            )
        if scale > 127 or scale < 64:
            raise ValueError(
                f"Global Average Pool scale must be between 64 and 127, got {scale}."
            )

        return bias.astype(np.int32), scale.astype(np.int8), shift.astype(np.int16)

    def mutate(self, op):
        new_op = super().mutate(op)
        subgraph = new_op.subgraph

        with self.using(new_op):
            # replace reduction_indices tensor with bias_scale_shift
            new_op.inputs[1].consumers.remove(new_op)
            new_op.inputs[1] = subgraph.create_tensor(
                f"{new_op.name}/bias_scale_shift",
                TensorType.INT8,
                shape=[7],
                consumers=[new_op],
            )
            new_op.inputs[1].buffer.data = b"".join(
                p.tobytes() for p in self._bias_scale_shift
            )

        return new_op
