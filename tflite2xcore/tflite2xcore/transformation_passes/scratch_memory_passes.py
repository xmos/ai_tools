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

    @property
    def _bias_scratch_size(self) -> int:
        _, Bv, Bl = self._op.inputs[2].shape
        return Bv * Bl * self._op.inputs[2].type.to_bytes()

    @property
    @abstractmethod
    def _weights_scratch_size(self) -> int:
        raise NotImplementedError()

    def mutate(self, op: Operator) -> None:
        with self.using(op):
            op.add_custom_options(
                mem=[self._weights_scratch_size, self._bias_scratch_size]
            )


class ScratchMemoryFullyConnectedPass(ScratchMemoryCalculationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_fc_deepin_anyout,)

    @property
    def _weights_scratch_size(self) -> int:
        Cout, Cin = self._op.inputs[1].shape

        custom_options = self._op.custom_options
        if "par" in custom_options:
            # get the min of threads or number of channel groups  # TODO: fix this
            i_cg = min(custom_options["par"]["th"], len(custom_options["par"]["cg"]))
            return Cin * (custom_options["par"]["cg"][i_cg - 1][1] + 1)
        else:
            return Cin * Cout


class ScratchMemoryConv2dPass(ScratchMemoryCalculationPass):
    MATCHING_OPCODES = (
        XCOREOpCodes.XC_conv2d_deep,
        XCOREOpCodes.XC_conv2d_shallowin,
        XCOREOpCodes.XC_conv2d_depthwise,
    )

    @property
    def _weights_scratch_size(self) -> int:
        _, _, _, Cin = self._op.inputs[0].shape
        if len(self._op.inputs[1].shape) == 4:
            _, Kh, Kw, _ = self._op.inputs[1].shape
        else:
            Kh, Kw, _ = self._op.inputs[1].shape

        custom_options = self._op.custom_options
        if "par" in custom_options:
            max_cg_size = max([cg[1] - cg[0] + 1 for cg in custom_options["par"]["cg"]])
        else:
            max_cg_size = CHANNEL_GROUP_SIZE

        return Cin * Kh * Kw * max_cg_size


class ScratchMemoryConv2d1x1Pass(ScratchMemoryCalculationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_conv2d_1x1,)

    @property
    def _weights_scratch_size(self) -> int:
        _, _, _, Cin = self._op.inputs[0].shape

        custom_options = self._op.custom_options
        if "par" in custom_options:
            max_cg_size = max([cg[1] - cg[0] + 1 for cg in custom_options["par"]["cg"]])
        else:
            max_cg_size = CHANNEL_GROUP_SIZE

        return Cin * max_cg_size
