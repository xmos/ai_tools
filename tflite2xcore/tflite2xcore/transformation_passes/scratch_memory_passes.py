# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

from abc import abstractmethod
from typing import Tuple

from tflite2xcore.xcore_schema import XCOREOpCodes, Operator, Tensor, Buffer
from tflite2xcore.parallelization import CHANNEL_GROUP_SIZE
from tflite2xcore.xcore_schema.operator_code import OperatorCode

from .transformation_passes import OperatorMatchingPass, TensorMatchingPass


class ScratchMemoryCalculationPass(OperatorMatchingPass):
    @property
    def _input(self):
        return self._op.inputs[0]

    @property
    def _weights(self):
        return self._op.inputs[1]

    @property
    def _biases(self):
        return self._op.inputs[2]

    @property
    @abstractmethod
    def MATCHING_OPCODES(self) -> Tuple[XCOREOpCodes, ...]:
        return tuple()

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and op.operator_code.code in self.MATCHING_OPCODES
            and "mem" not in op.custom_options
        )

    @property
    def _bias_scratch_size(self) -> int:
        _, Bv, Bl = self._biases.shape
        return Bv * Bl * self._biases.type.sizeof()

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
    MATCHING_OPCODES = (XCOREOpCodes.XC_fc,)

    @property
    def _weights_scratch_size(self) -> int:
        Cout, Cin = self._weights.shape

        custom_options = self._op.custom_options
        if "par" in custom_options:
            # NOTE: number of channel groups is at least number of threads
            i_cg = custom_options["par"]["th"]
            return Cin * (custom_options["par"]["cg"][i_cg - 1][1] + 1)
        else:
            return Cin * Cout

    @property
    def _bias_scratch_size(self) -> int:
        _, Bv, Bl = self._biases.shape

        custom_options = self._op.custom_options
        if "par" in custom_options:
            # NOTE: number of channel groups is at least number of threads
            i_cg = custom_options["par"]["th"]
            return Bv * Bl * self._biases.type.sizeof() * i_cg
        else:
            return Bv * Bl * self._biases.type.sizeof()


class Conv2dScratchMemoryCalculationPass(ScratchMemoryCalculationPass):
    @property
    @abstractmethod
    def _kernel_size(self) -> Tuple[int, int]:
        raise NotImplementedError()

    @property
    def _max_channel_group_size(self):
        custom_options = self._op.custom_options
        if "par" in custom_options:
            max_cg_size = max([cg[1] - cg[0] + 1 for cg in custom_options["par"]["cg"]])
        else:
            max_cg_size = CHANNEL_GROUP_SIZE

        return max_cg_size

    @property
    def _weights_scratch_size(self) -> int:
        _, _, _, Cin = self._input.shape
        Kh, Kw = self._kernel_size
        return Cin * Kh * Kw * self._max_channel_group_size


class ScratchMemoryConv2dPass(Conv2dScratchMemoryCalculationPass):
    MATCHING_OPCODES = (
        XCOREOpCodes.XC_conv2d_deep,
        XCOREOpCodes.XC_conv2d_shallowin,
    )

    @property
    def _kernel_size(self) -> Tuple[int, int]:
        return self._weights.shape[1:3]


class ScratchMemoryDepthwiseConv2dPass(Conv2dScratchMemoryCalculationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_conv2d_depthwise,)

    @property
    def _kernel_size(self) -> Tuple[int, int]:
        return self._weights.shape[0:2]

    @property
    def _weights_scratch_size(self) -> int:
        Kh, Kw = self._kernel_size
        return Kh * Kw * self._max_channel_group_size


class ScratchMemoryConv2d1x1Pass(Conv2dScratchMemoryCalculationPass):
    MATCHING_OPCODES = (XCOREOpCodes.XC_conv2d_1x1,)

    @property
    def _kernel_size(self) -> Tuple[int, int]:
        return 1, 1


class InsertExternalMemoryFetchPass(TensorMatchingPass):
    MATCHING_OPCODES = (
        XCOREOpCodes.XC_bconv2d_bin,
        XCOREOpCodes.XC_bconv2d_bin_DI,
        XCOREOpCodes.XC_bconv2d_int8,
        XCOREOpCodes.XC_bconv2d_int8_DIDO,
    )

    def match(self, tensor: Tensor) -> bool:
        if super().match(tensor) and tensor.is_constant and len(tensor.consumers) == 1:
            consumer = tensor.consumers[0]
            return consumer.operator_code.code in self.MATCHING_OPCODES
        return False

    def mutate(self, tensor: Tensor) -> None:
        subgraph = tensor.subgraph

        # clone the constant tensor, rename the original, remove old buffer
        new_tensor = subgraph.clone_tensor(tensor)
        tensor.name = subgraph.make_unique_tensor_name(tensor.name)
        tensor.buffer.owners = []
        tensor.buffer = Buffer(tensor.model, owners=[tensor])

        # create and insert fetch op
        fetch_op = subgraph.create_operator(
            OperatorCode(XCOREOpCodes.XC_fetch), inputs=[new_tensor], outputs=[tensor]
        )
        fetch_op.name = f"{new_tensor.name}/fetch"
        subgraph.insert_operator(tensor.consumers[0], fetch_op)
