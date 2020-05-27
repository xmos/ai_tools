# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np

from tflite2xcore.xcore_schema import (
    TensorType,
    BuiltinOpCodes,
    OperatorCode,
    XCOREOpCodes,
)
from tflite2xcore.utils import VE, WORD_SIZE
from .transformation_passes import ReplaceQuantizedOperatorPass, OperatorMatchingPass
from tflite2xcore.execution_planning import ChannelGroupSlicePlanner, SlicePlanner


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
                    and self._fused_activation == "NONE"
                    and self._input.shape[3] % 4 == 0
                )

        return False

    def mutate(self, op):
        new_op = super().mutate(op)

        with self.using(op):
            new_op.add_custom_options(
                stride=list(self._strides), pool=list(self._pool_size)
            )


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
                return self._padding == "VALID"

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
                return self._padding == "VALID"

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
                reduction_dims = self._op.inputs[1].numpy
                return (
                    len(reduction_dims) == 2
                    and np.all(reduction_dims == [1, 2])
                    and self._input.shape[3] % WORD_SIZE == 0
                )

        return False

    @property
    def _bias_scale_shift(self):
        num_pixels = self._input.shape[1] * self._input.shape[2]
        rescaling = (
            self._input.quantization["scale"][0] / self._output.quantization["scale"][0]
        )
        multiplier = rescaling / num_pixels

        scale = np.round(multiplier * 2 ** (7 - np.ceil(np.log2(multiplier))))
        shift = np.round(np.log2(scale / multiplier))
        bias = np.round(
            scale
            * (
                self._output.quantization["zero_point"][0] / multiplier
                - self._input.quantization["zero_point"][0] * num_pixels
            )
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
                f"{new_op.name}/bias_scale_shift",
                TensorType.INT8,
                shape=[7],
                consumers=[new_op],
            )
            new_op.inputs[1].buffer.data = np.frombuffer(
                b"".join(p.tostring() for p in self._bias_scale_shift), dtype=np.int8
            )
            subgraph.remove_tensor(old_tensor)

        return new_op


class PlanGlobalAveragePool2DPass(OperatorMatchingPass):
    def __init__(self, *args, num_threads=None, forced=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_threads = num_threads or 1
        assert isinstance(self.max_threads, int)
        assert self.max_threads > 0
        self.forced = forced
        self.plan_threads = None

    def run(self, *args, **kwargs):
        if self.max_threads == 1:
            self.logger.debug(f"Skipping pass b/c num_threads={self.max_threads}")
            return 0
        else:
            return super().run(*args, **kwargs)

    def match(self, op):
        if (
            super().match(op)
            and op.operator_code.code == XCOREOpCodes.XC_avgpool2d_global
        ):
            return not self.plan_threads

    def mutate(self, op):
        _, Cout = op.outputs[0].shape
        assert int(Cout) == Cout
        planner = ChannelGroupSlicePlanner(
            int(Cout), num_threads=self.max_threads, forced=self.forced
        )
        plan = planner.find_optimal_plan()
        self.plan_threads = plan.num_threads

        if self.plan_threads > 1:
            op.add_custom_options(par=plan.to_dict())


class PlanPooling2DPass(OperatorMatchingPass):
    def __init__(self, *args, num_threads=None, forced=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_threads = num_threads or 1
        assert isinstance(self.num_threads, int)
        assert self.num_threads > 0
        self.forced = forced
        self.plan_threads = None

    MATCHING_OPCODES = (XCOREOpCodes.XC_maxpool2d, XCOREOpCodes.XC_avgpool2d)

    def run(self, *args, **kwargs):
        if self.num_threads == 1:
            self.logger.debug(f"Skipping pass b/c num_threads={self.num_threads}")
            return 0
        else:
            return super().run(*args, **kwargs)

    def match(self, op):
        if super().match(op) and op.operator_code.code in self.MATCHING_OPCODES:
            return not self.plan_threads

    def mutate(self, op):
        _, height, width, Cout = op.outputs[0].shape
        assert int(height) == height
        assert int(width) == width
        planner = SlicePlanner(
            int(Cout),
            int(height),
            int(width),
            num_threads=self.num_threads,
            forced=self.forced,
        )
        plan = planner.find_optimal_plan()
        self.plan_threads = plan.num_threads

        if self.plan_threads > 1:
            op.add_custom_options(plan=plan.to_dict())
