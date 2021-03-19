# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import numpy as np
from abc import abstractmethod
from typing import Tuple, Optional, Any

from tflite2xcore.xcore_schema import XCOREOpCodes, Operator
from tflite2xcore.parallelization import (
    ParallelizationPlanner,
    SlicePlanner,
    ChannelGroupSlicePlanner,
    ElementWisePlanner,
)
from tflite2xcore.utils import WORD_SIZE_BITS, WORD_SIZE_BYTES, ACC_PERIOD_INT8

from .transformation_passes import OperatorMatchingPass


class ParallelizationPass(OperatorMatchingPass):
    FIXED_COST_PER_THREAD = 0

    @property
    @abstractmethod
    def MATCHING_OPCODES(self) -> Tuple[XCOREOpCodes, ...]:
        return tuple()

    def __init__(
        self,
        *args: Any,
        num_threads: Optional[int] = None,
        forced: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        num_threads = num_threads or 1
        assert num_threads > 0
        self._planner_args = dict(
            forced=forced,
            num_threads=num_threads,
            fixed_cost_per_thread=self.FIXED_COST_PER_THREAD,
        )

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


class ParallelizeElementWisePass(ParallelizationPass):
    JOB_SIZE_ALIGNMENT = WORD_SIZE_BYTES

    @property
    def _num_elements(self) -> int:
        return int(np.prod(self._op.outputs[0].shape[1:]))

    @property
    def _planner(self) -> ElementWisePlanner:
        return ElementWisePlanner(
            self._num_elements, alignment=self.JOB_SIZE_ALIGNMENT, **self._planner_args
        )


class ParallelizeLUTPass(ParallelizeElementWisePass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_lookup_8,)
    FIXED_COST_PER_THREAD = 10
    JOB_SIZE_ALIGNMENT = 1


class ParallelizeAddPass(ParallelizeElementWisePass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_add_8,)
    FIXED_COST_PER_THREAD = 100


class ParallelizeChannelWisePass(ParallelizeElementWisePass):
    FIXED_COST_PER_THREAD = 0
    JOB_SIZE_ALIGNMENT = ACC_PERIOD_INT8

    @property
    def _num_elements(self) -> int:
        num_channels = self._op.outputs[0].shape[-1]
        assert num_channels % WORD_SIZE_BYTES == 0
        return num_channels


class ParallelizeGlobalAveragePool2DPass(ParallelizeChannelWisePass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_avgpool2d_global,)


class ChannelGroupParallelizationPass(ParallelizationPass):
    @property
    def _planner(self) -> ChannelGroupSlicePlanner:
        output_shape = self._op.outputs[0].shape
        Cout = np.prod(output_shape[1:])  # works even if output is (1, 1, 1, Cout)
        assert output_shape[-1] == Cout
        return ChannelGroupSlicePlanner(Cout, **self._planner_args)


class SpatialParallelizationPass(ParallelizationPass):
    @property
    def _cout(self) -> int:
        return self._op.outputs[0].shape[3]

    @property
    def _planner(self) -> SlicePlanner:
        _, height, width, _ = self._op.outputs[0].shape
        return SlicePlanner(self._cout, height, width, **self._planner_args)


class ParallelizeFullyConnectedPass(ChannelGroupParallelizationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_fc,)


class ParallelizeRequant16To8Pass(ChannelGroupParallelizationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_requantize_16_to_8,)


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
