# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import numpy as np
from abc import abstractmethod
from typing import Tuple

from tflite2xcore.xcore_model import Operator
from tflite2xcore.xcore_schema import XCOREOpCodes
from tflite2xcore.parallelization import (
    ParallelizationPlanner,
    SlicePlanner,
    ChannelGroupSlicePlanner,
)
from tflite2xcore.utils import WORD_SIZE_BITS

from .transformation_passes import OperatorMatchingPass


class ParallelizationPass(OperatorMatchingPass):
    @property
    @abstractmethod
    def MATCHING_OPCODES(self) -> Tuple[XCOREOpCodes, ...]:
        return tuple()

    def __init__(
        self, *args, num_threads: int = None, forced: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_threads = num_threads or 1
        assert isinstance(self.num_threads, int)
        assert self.num_threads > 0
        self.forced = forced

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and op.operator_code.code in self.MATCHING_OPCODES
            and "par" not in op.custom_options
        )

    @property
    @abstractmethod
    def _planner(self) -> ParallelizationPlanner:
        raise NotImplementedError()

    def mutate(self, op: Operator) -> None:
        with self.using(op):
            op.add_custom_options(par=self._planner.find_optimal_plan().to_dict())


class ChannelGroupParallelizationPass(ParallelizationPass):
    @property
    def _planner(self) -> ChannelGroupSlicePlanner:
        output_shape = self._op.outputs[0].shape
        Cout = np.prod(output_shape[1:])  # works even if output is (1, 1, 1, Cout)
        assert output_shape[-1] == Cout
        return ChannelGroupSlicePlanner(
            int(Cout), num_threads=self.num_threads, forced=self.forced
        )


class SpatialParallelizationPass(ParallelizationPass):
    @property
    def _cout(self) -> int:
        return int(self._op.outputs[0].shape[3])

    @property
    def _planner(self) -> SlicePlanner:
        _, height, width, _ = self._op.outputs[0].shape
        return SlicePlanner(
            self._cout, height, width, num_threads=self.num_threads, forced=self.forced,
        )


class ParallelizeFullyConnectedPass(ChannelGroupParallelizationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_fc,)


class ParallelizeRequant16To8Pass(ChannelGroupParallelizationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_requantize_16_to_8,)


class ParallelizeGlobalAveragePool2DPass(ChannelGroupParallelizationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_avgpool2d_global,)


class ParallelizeConv2dPass(SpatialParallelizationPass):
    MATCHING_OPCODES = (
        XCOREOpCodes.XC_conv2d_shallowin,
        XCOREOpCodes.XC_conv2d_deep,
        XCOREOpCodes.XC_conv2d_1x1,
    )


class ParallelizeBConv2dInt8Pass(SpatialParallelizationPass):
    MATCHING_OPCODES = (
        XCOREOpCodes.XC_bconv2d_int8,
        XCOREOpCodes.XC_bconv2d_int8_DIDO,
    )

    def mutate(self, op: Operator) -> None:
        with self.using(op):
            par = self._planner.find_optimal_plan().to_dict()
            par.pop("cg")
            op.add_custom_options(par=par)


class ParallelizeBConv2dBinPass(ParallelizeBConv2dInt8Pass):
    MATCHING_OPCODES = (
        XCOREOpCodes.XC_bconv2d_bin,
        XCOREOpCodes.XC_bconv2d_bin_DI,
    )

    @property
    def _cout(self) -> int:
        return super()._cout * WORD_SIZE_BITS


class ParallelizeDepthwiseConv2dPass(SpatialParallelizationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_conv2d_depthwise,)


class ParallelizePooling2DPass(SpatialParallelizationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_maxpool2d, XCOREOpCodes.XC_avgpool2d)
