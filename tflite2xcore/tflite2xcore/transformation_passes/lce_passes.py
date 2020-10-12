# Copyright (c) 2020, XMOS Ltd, All rights reserved
import numpy as np
from typing import Tuple

from tflite2xcore.utils import (
    WORD_SIZE_BITS,
    VECTOR_SIZE_BITS,
    ACC_PERIOD,
    WORD_SIZE,
    calculate_same_padding,
)
from tflite2xcore.xcore_model import Operator
from tflite2xcore.xcore_schema import (
    Padding,
    TensorType,
    ExternalOpCodes,
    XCOREOpCodes,
    OperatorCode,
    BuiltinOpCodes,
)

from .transformation_passes import (
    OperatorMatchingPass,
    ReplaceQuantizedOperatorPass,
)
from .conv2d_passes import ReplaceConv2DPass


class ReplaceBconv2DPass(ReplaceConv2DPass):
    @property
    def matching_opcode(self) -> ExternalOpCodes:
        return ExternalOpCodes.add_new_opcode("LceBconv2d")

    @property
    def matching_input_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def matching_biases_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def matching_weights_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def _strides(self) -> Tuple[int, int]:
        options = self._op.custom_options
        return options["stride_height"], options["stride_width"]

    @property
    def _dilation(self) -> Tuple[int, int]:
        options = self._op.custom_options
        return options["dilation_height_factor"], options["dilation_width_factor"]

    @property
    def _padding(self) -> Padding:
        return self._op.custom_options["padding"]

    @property
    def _input_channels(self) -> int:
        return self._op.custom_options["channels_in"]

    @property
    def _output_channels(self) -> int:
        return self._weights.shape[0]

    def match(self, op: Operator) -> bool:
        if super().match(op):
            with self.using(op):
                if self._input_channels != self._weights.shape[3] * WORD_SIZE_BITS:
                    self.logger.warning(
                        f"Found {self.matching_opcode} operator "
                        f"with {self._input_channels} input channels "
                        f"(not a multiple of {WORD_SIZE_BITS})."
                    )
                elif self._output_channels % WORD_SIZE != 0:
                    self.logger.warning(
                        f"Found {self.matching_opcode} operator "
                        f"with {self._output_channels} output channels "
                        f"(not a multiple of {WORD_SIZE})"
                    )
                else:
                    return True

        return False

    def mutate(self, op: Operator) -> None:
        new_op = super().mutate(op)
        with self.using(op):
            new_op.add_custom_options(
                stride=self._strides, padding=self._padding,
            )
        return new_op


class ReplaceBconv2DInt8Pass(ReplaceBconv2DPass):
    @property
    def matching_biases_type(self) -> TensorType:
        return TensorType.FLOAT32

    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_bconv2d_int8)

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and len(op.inputs) == 4
            and op.inputs[3].type is self.matching_biases_type
        )


class ReplaceBconv2DInt8DeepInDeepOutPass(ReplaceBconv2DInt8Pass):
    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_bconv2d_int8_DIDO)

    def match(self, op: Operator) -> bool:
        with self.using(op):
            return (
                super().match(op)
                and self._input_channels % VECTOR_SIZE_BITS == 0
                and self._output_channels % ACC_PERIOD == 0
            )


class ReplaceBconv2DBitpackedPass(ReplaceBconv2DPass):
    @property
    def matching_output_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_bconv2d_bin)

    def match(self, op: Operator) -> bool:
        if super().match(op) and len(op.inputs) == 3:
            with self.using(op):
                if self._output_channels % WORD_SIZE_BITS == 0:
                    return True
                self.logger.warning(
                    f"Found {self.matching_opcode} operator with bitpacked output "
                    f"and {self._output_channels} output channels "
                    f"(not a multiple of {WORD_SIZE_BITS})"
                )
        return False


class ReplaceBconv2DBitpackedDeepInPass(ReplaceBconv2DBitpackedPass):
    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_bconv2d_bin_DI)

    def match(self, op: Operator) -> bool:
        with self.using(op):
            return super().match(op) and self._input_channels % VECTOR_SIZE_BITS == 0


class ReplaceLceQuantizePass(ReplaceQuantizedOperatorPass):
    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_bsign_8)

    @property
    def matching_opcode(self) -> ExternalOpCodes:
        return ExternalOpCodes.add_new_opcode("LceQuantize")

    @property
    def matching_output_type(self) -> TensorType:
        return TensorType.INT32

    def match(self, op: Operator) -> bool:
        if super().match(op):
            input_shape = op.inputs[0].shape
            if len(input_shape) == 4 and input_shape[3] % WORD_SIZE_BITS == 0:
                return True
            self.logger.warning(
                f"Found LceQuantize with illegal input shape {input_shape}"
            )
        return False


# Split out padding to a separate op from BConv
# TODO: this currently only matches with XC_bconv2d_*
# but going forward might like to extend this to other conv ops
# and make it a more general pass for all convolutions.
class LegalizeXCBconv2DPaddingPass(OperatorMatchingPass):
    @property
    def _strides(self) -> Tuple[int, int]:
        return self._op.custom_options["stride"]

    @property
    def _padding(self) -> Padding:
        return self._op.custom_options["padding"]

    MATCHING_OPCODES = (
        XCOREOpCodes.XC_bconv2d_int8,
        XCOREOpCodes.XC_bconv2d_int8_DIDO,
        XCOREOpCodes.XC_bconv2d_bin,
        XCOREOpCodes.XC_bconv2d_bin_DI,
    )

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and op.operator_code.code in self.MATCHING_OPCODES
            and "padding" in op.custom_options
        )

    def mutate(self, op: Operator) -> Operator:
        padding = Padding(op.custom_options.pop("padding"))
        if padding is Padding.VALID:
            return op

        subgraph = op.subgraph
        old_input = op.inputs[0]

        # calculate paddings
        with self.using(op):
            input_and_strides = old_input.shape[1:3], self._strides

        paddings = np.int32(
            [
                (0, 0),
                *calculate_same_padding(*input_and_strides, op.inputs[1].shape[1:3]),
                (0, 0),
            ]
        )

        # Construct paddings parameter tensor and padded input tensor
        padding_tensor = subgraph.create_tensor(
            f"{op.name}/paddings", TensorType.INT32, shape=paddings.shape
        )
        padding_tensor.buffer.data = paddings

        padded_shape = tuple(
            int(size + sum(pads)) for size, pads in zip(old_input.shape, paddings)
        )
        padded_input_tensor = subgraph.create_tensor(
            f"{op.name}/input", TensorType.INT32, shape=padded_shape, consumers=[op],
        )

        # create new PAD op and inject it before the convolution
        pad_op = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.PAD),
            inputs=[old_input, padding_tensor],
            outputs=[padded_input_tensor],
        )
        subgraph.insert_operator(op, pad_op)

        # Cut connection from old input to the op
        old_input.consumers.remove(op)
        op.inputs[0] = padded_input_tensor

        return op
