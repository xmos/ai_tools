# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import abstractmethod
from typing import Tuple

from tflite2xcore.xcore_model import Operator
from tflite2xcore.xcore_schema import XCOREOpCodes
from tflite2xcore.parallelization import CHANNEL_GROUP_SIZE
from .transformation_passes import OperatorMatchingPass


class ScratchMemoryCalculationPass(OperatorMatchingPass):
    @property
    @abstractmethod
    def MATCHING_OPCODES(self) -> Tuple[XCOREOpCodes]:
        return tuple()

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and op.operator_code.code in self.MATCHING_OPCODES
            and "mem" not in op.custom_options
        )


class ScratchMemoryFullyConnectedPass(ScratchMemoryCalculationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_fc_deepin_anyout,)

    def mutate(self, op):
        Cout, Cin = op.inputs[1].shape
        _, Bv, Bl = op.inputs[2].shape

        if "par" in op.custom_options:
            # get the min of threads or number of channel groups
            i_cg = min(
                op.custom_options["par"]["th"], len(op.custom_options["par"]["cg"])
            )
            weights_scratch_size = Cin * (
                op.custom_options["par"]["cg"][i_cg - 1][1] + 1
            )
        else:
            weights_scratch_size = Cin * Cout

        bias_scratch_size = Bv * Bl * op.inputs[2].type.to_bytes()

        op.add_custom_options(mem=[weights_scratch_size, bias_scratch_size])


class ScratchMemoryConv2dPass(ScratchMemoryCalculationPass):
    MATCHING_OPCODES = (
        XCOREOpCodes.XC_conv2d_deep,
        XCOREOpCodes.XC_conv2d_shallowin,
        XCOREOpCodes.XC_conv2d_depthwise,
    )

    def mutate(self, op):
        _, _, _, Cin = op.inputs[0].shape
        if len(op.inputs[1].shape) == 4:
            _, Kh, Kw, _ = op.inputs[1].shape
        else:
            Kh, Kw, _ = op.inputs[1].shape

        _, Bv, Bl = op.inputs[2].shape

        if "par" in op.custom_options:
            max_cg_size = max(
                [cg[1] - cg[0] + 1 for cg in op.custom_options["par"]["cg"]]
            )
        else:
            max_cg_size = CHANNEL_GROUP_SIZE

        weights_scratch_size = Cin * Kh * Kw * max_cg_size
        bias_scratch_size = Bv * Bl * op.inputs[2].type.to_bytes()

        op.add_custom_options(mem=[weights_scratch_size, bias_scratch_size])


class ScratchMemoryConv2d1x1Pass(ScratchMemoryCalculationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_conv2d_1x1,)

    def mutate(self, op):
        _, _, _, Cin = op.inputs[0].shape
        _, Bv, Bl = op.inputs[2].shape

        if "par" in op.custom_options:
            max_cg_size = max(
                [cg[1] - cg[0] + 1 for cg in op.custom_options["par"]["cg"]]
            )
        else:
            max_cg_size = CHANNEL_GROUP_SIZE

        weights_scratch_size = Cin * max_cg_size
        bias_scratch_size = Bv * Bl * op.inputs[2].type.to_bytes()

        op.add_custom_options(mem=[weights_scratch_size, bias_scratch_size])
