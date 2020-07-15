# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import abstractmethod

from tflite2xcore.xcore_schema import XCOREOpCodes
from tflite2xcore.parallelization import SlicePlanner, ChannelGroupSlicePlanner

from .transformation_passes import OperatorMatchingPass


class ParallelizationPass(OperatorMatchingPass):
    @property
    @abstractmethod
    def MATCHING_OPCODES(self):
        return tuple()

    def __init__(self, *args, num_threads=None, forced=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_threads = num_threads or 1
        assert isinstance(self.num_threads, int)
        assert self.num_threads > 0
        self.forced = forced

    def match(self, op):
        return (
            super().match(op)
            and op.operator_code.code in self.MATCHING_OPCODES
            and "par" not in op.custom_options
        )


class ChannelGroupParallelizationPass(ParallelizationPass):
    def mutate(self, op):
        _, Cout = op.outputs[0].shape
        planner = ChannelGroupSlicePlanner(
            int(Cout), num_threads=self.num_threads, forced=self.forced
        )
        plan = planner.find_optimal_plan()
        plan.num_threads = min(
            plan.num_threads, len(plan.changrp_slices)
        )  # TODO: fix this

        op.add_custom_options(par=plan.to_dict())


class SpatialParallelizationPass(ParallelizationPass):
    def mutate(self, op):
        _, height, width, Cout = op.outputs[0].shape
        planner = SlicePlanner(
            int(Cout),
            int(height),
            int(width),
            num_threads=self.num_threads,
            forced=self.forced,
        )
        plan = planner.find_optimal_plan()

        op.add_custom_options(par=plan.to_dict())


class ParallelizeFullyConnectedPass(ChannelGroupParallelizationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_fc_deepin_anyout,)


class ParallelizeRequant16To8Pass(ChannelGroupParallelizationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_requantize_16_to_8,)


class ParallelizeGlobalAveragePool2DPass(ChannelGroupParallelizationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_avgpool2d_global,)


class ParallelizeConv2dPass(SpatialParallelizationPass):
    MATCHING_OPCODES = (
        XCOREOpCodes.XC_conv2d_shallowin,
        XCOREOpCodes.XC_conv2d_deep,
        XCOREOpCodes.XC_conv2d_1x1,
        XCOREOpCodes.XC_conv2d_depthwise,
    )


class ParallelizePooling2DPass(SpatialParallelizationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_maxpool2d, XCOREOpCodes.XC_avgpool2d)
