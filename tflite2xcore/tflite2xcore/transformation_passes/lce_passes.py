# Copyright (c) 2020, XMOS Ltd, All rights reserved
from typing import Any

import numpy as np

from tflite2xcore.utils import WORD_SIZE_BITS, VECTOR_SIZE_BITS
from .transformation_passes import OperatorMatchingPass
from tflite2xcore.xcore_model import Operator
from tflite2xcore.xcore_schema import (
    Padding,
    TensorType,
    BuiltinOpCodes,
    ExternalOpCodes,
    XCOREOpCodes,
    OperatorCode,
    BuiltinOptions,
)


class LceConv2dPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:

        if not super().match(op):
            return False

        try:
            if op.operator_code.code is not ExternalOpCodes.LceBconv2d:
                return False
        except AttributeError:
            return False

        options = op.custom_options

        strides = (options["stride_height"], options["stride_width"])
        dilations = (
            options["dilation_height_factor"],
            options["dilation_width_factor"],
        )

        padding = options["padding"]
        weights = op.inputs[1]

        return (
            len(op.inputs) == 3
            and strides == (1, 1)
            and dilations == (1, 1)
            and weights.shape[0] % WORD_SIZE_BITS == 0  # Cout
            and (weights.shape[3] * WORD_SIZE_BITS) % VECTOR_SIZE_BITS == 0  # Cin
            and weights.type is TensorType.INT32
            and op.inputs[0].type is TensorType.INT32
        )


# Replace LCEBconv2D with XC_BConv2D
class ReplaceLceBconv2DPass(LceConv2dPass):
    def __init__(
        self, output_tensor_type: TensorType, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._output_tensor_type = output_tensor_type

    def match(self, op: Operator) -> bool:

        return (
            super().match(op)
            and len(op.inputs) == 3
            and op.inputs[2].type is TensorType.INT32
            and op.outputs[0].type is self._output_tensor_type
        )

    def mutate(self, op: Operator) -> None:

        subgraph = op.subgraph

        # Note, it is risky to modify an Op in place - create a new op and remove old one
        if self._output_tensor_type is TensorType.INT8:
            new_op_code = XCOREOpCodes.XC_bconv2d_int8_out
        else:
            new_op_code = XCOREOpCodes.XC_bconv2d_bin_out

        bconv_op = subgraph.create_operator(
            OperatorCode(opcode=new_op_code),
            inputs=op.inputs,
            outputs=op.outputs,
            custom_options=op.custom_options,
        )

        subgraph.insert_operator(op, bconv_op)

        subgraph.remove_operator(op)


# Replace LCEQuantize with XC_Bsign8
class ReplaceLceQuantizePass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:

        try:
            if op.operator_code.code is not ExternalOpCodes.LceQuantize:
                return False
        except AttributeError:
            return False

        return (
            super().match(op)
            and len(op.inputs) == 1
            and len(op.outputs) == 1
            and op.outputs[0].type is TensorType.INT32
            and op.inputs[0].type is TensorType.INT8
        )

    def mutate(self, op: Operator) -> None:

        subgraph = op.subgraph

        bsign_op = subgraph.create_operator(
            OperatorCode(opcode=XCOREOpCodes.XC_bsign_8),
            inputs=op.inputs,
            outputs=op.outputs,
            custom_options=op.custom_options,
        )
        subgraph.insert_operator(op, bsign_op)
        subgraph.remove_operator(op)


# Split out padding to a separate op from BConv
# Note, this currently only matches with BConv but going forward might like to extend this to other conv ops
# and make it a general pass. Bconv only supports SAME and VALID spacial padding
class SplitPaddingFromConvPass(LceConv2dPass):
    @staticmethod
    def _calc_pad(stride, input_size, kernel_size):
        return int(np.ceil(((stride - 1) * input_size - stride + kernel_size) / 2))

    def match(self, op: Operator) -> bool:

        return (
            super().match(op) and Padding(op.custom_options["padding"]) is Padding.SAME
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
        padding_tb = SplitPaddingFromConvPass._calc_pad(strides[0], height, K_h)
        padding_lr = SplitPaddingFromConvPass._calc_pad(strides[1], width, K_w)
        paddings = [[0, 0], [padding_tb, padding_tb], [padding_lr, padding_lr], [0, 0]]
        padding_tensor = subgraph.create_tensor(
            f"{op.name}/paddings", TensorType.INT32, shape=[4, 2]
        )
        padding_tensor.buffer.data = np.int32(paddings)

        pad_output_shape = tuple(
            input_dim + sum(pad) for input_dim, pad in zip(old_input.shape, paddings)
        )

        pad_output_tensor = subgraph.create_tensor(
            f"{op.name}/input",
            TensorType.INT32,
            shape=pad_output_shape,
            consumers=[op],
        )

        # Note, going foward a builtin.PAD will be inserted, to be replaced by a later pass
        pad_op = subgraph.create_operator(
            OperatorCode(opcode=XCOREOpCodes.XC_pad),
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
