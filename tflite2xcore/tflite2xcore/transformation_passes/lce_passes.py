# Copyright (c) 2020, XMOS Ltd, All rights reserved
import numpy as np
from typing import Tuple

from tflite2xcore.utils import (
    WORD_SIZE_BITS,
    VECTOR_SIZE_BITS,
    ACC_PERIOD,
    calculate_same_padding,
    calculate_same_output_size,
)
from tflite2xcore.xcore_model import Operator
from tflite2xcore.xcore_schema import (
    Padding,
    TensorType,
    ExternalOpCodes,
    XCOREOpCodes,
    OperatorCode,
    BuiltinOptions,
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

    def match(self, op: Operator) -> bool:
        # other versions of the LCE op can have different number of inputs
        if super().match(op) and len(op.inputs) == 3:
            with self.using(op):
                inferred_input_channels = self._weights.shape[3] * WORD_SIZE_BITS
                return (
                    inferred_input_channels == self._input_channels
                    # number of input channels must be multiple of 256
                    and inferred_input_channels % VECTOR_SIZE_BITS == 0
                )

        return False

    def mutate(self, op: Operator) -> None:
        new_op = super().mutate(op)
        with self.using(op):
            new_op.add_custom_options(
                stride=self._strides, padding=self._padding,
            )
        return new_op


class ReplaceBconv2DInt8OutPass(ReplaceBconv2DPass):
    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_bconv2d_int8_out)

    def match(self, op: Operator) -> bool:
        with self.using(op):
            return super().match(op) and self._weights.shape[0] % ACC_PERIOD == 0


class ReplaceBconv2DBitpackedOutPass(ReplaceBconv2DPass):
    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_bconv2d_bin_out)

    @property
    def matching_output_type(self) -> TensorType:
        return TensorType.INT32


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
# Note, this currently only matches with BConv but going forward might like to extend this to other conv ops
# and make it a general pass. Bconv only supports SAME and VALID spacial padding
class SplitPaddingFromConvPass(OperatorMatchingPass):
    MATCHING_OPCODES = (
        XCOREOpCodes.XC_bconv2d_int8_out,
        XCOREOpCodes.XC_bconv2d_bin_out,
    )

    @staticmethod
    def _calc_pad(stride, input_size, kernel_size):
        return int(np.ceil(((stride - 1) * input_size - stride + kernel_size) / 2))

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and op.operator_code.code in self.MATCHING_OPCODES
            and Padding(op.custom_options["padding"]) is Padding.SAME
        )

    def mutate(self, op: Operator) -> None:
        subgraph = op.subgraph

        height, width = op.inputs[0].shape[1:3]

        C_out, K_h, K_w, C_in = op.inputs[1].shape

        # Cut connection from old input to the op
        old_input = op.inputs[0]
        old_input.consumers.remove(op)

        strides = (
            op.custom_options["stride_height"],
            op.custom_options["stride_width"],
        )

        # Construct paddings input tensor for PAD op
        padding_tb = self._calc_pad(strides[0], height, K_h)
        padding_lr = self._calc_pad(strides[1], width, K_w)
        paddings = [[pad] * 2 for pad in (0, padding_tb, padding_lr, 0)]
        padding_tensor = subgraph.create_tensor(
            f"{op.name}/paddings", TensorType.INT32, shape=[4, 2]
        )
        padding_tensor.buffer.data = np.int32(paddings)

        pad_output_shape = tuple(
            input_dim + sum(pad) for input_dim, pad in zip(old_input.shape, paddings)
        )

        pad_output_tensor = subgraph.create_tensor(
            f"{op.name}/input", TensorType.INT32, shape=pad_output_shape
        )

        pad_op = subgraph.create_operator(
            OperatorCode(XCOREOpCodes.XC_pad),
            inputs=[old_input, padding_tensor],
            outputs=[pad_output_tensor],
        )

        op.inputs[0] = pad_op.outputs[0]

        subgraph.insert_operator(op, pad_op)

        # Pass on pad values from conv to pad op
        pad_op.custom_options["pad_values"] = op.custom_options["pad_values"]

        bytes_per_pixel = C_in * 4

        pad_op.custom_options["bytes_per_pixel"] = int(bytes_per_pixel)

        # Change padding of Bconv from SAME to VALID
        op.custom_options["padding"] = Padding.VALID
