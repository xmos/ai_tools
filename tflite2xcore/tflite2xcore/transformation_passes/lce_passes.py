# Copyright (c) 2020, XMOS Ltd, All rights reserved
from typing import Any

import numpy as np

from tflite2xcore.utils import WORD_SIZE_BITS, VECTOR_SIZE_BITS
from .transformation_passes import (
    OperatorMatchingPass,
    LegalizeWeightBiasPass,
    LegalizeXCWeightBiasPass,
)
from tflite2xcore.xcore_model import Operator, Tensor
from tflite2xcore.xcore_schema import (
    Padding,
    TensorType,
    BuiltinOpCodes,
    ExternalOpCodes,
    XCOREOpCodes,
    OperatorCode,
    BuiltinOptions,
)


def SupportedBconv2DOp(op: Operator) -> bool:

    if op.operator_code.code is not ExternalOpCodes.LceBconv2d:
        return False

    options = op.custom_options

    strides = (options["stride_height"], options["stride_width"])
    dilations = (
        options["dilation_height_factor"],
        options["dilation_width_factor"],
    )

    padding = Padding.from_TfLitePadding(options["padding"])

    weights = op.inputs[1]

    return (
        strides == (1, 1)
        and dilations == (1, 1)
        and weights.shape[0] % WORD_SIZE_BITS == 0  # Cout
        and (weights.shape[3] * WORD_SIZE_BITS) % VECTOR_SIZE_BITS == 0  # Cin
        and weights.type is TensorType.INT32
        and op.inputs[0].type in (TensorType.INT8, TensorType.INT32)
    )


class LceConv2dPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:

        if not super().match(op):
            return False

        return SupportedBconv2DOp(op)


class CanonicalizeLceBconv2DPass(LceConv2dPass):
    def match(self, op: Operator) -> bool:

        if not super().match(op):
            return False

        return len(op.inputs) == 4

    def mutate(self, op: Operator) -> None:
        op.inputs[2].consumers.remove(op)
        op.inputs[3].consumers.remove(op)
        op.inputs = op.inputs[:2]


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
            and len(op.inputs) == 2
            and op.outputs[0].type is self._output_tensor_type
        )

    # TODO replace op properly when relevant operator API is available, for now just replace code
    # to allow for basic testing of pass
    def mutate(self, op: Operator) -> None:
        if self._output_tensor_type is TensorType.INT8:
            op.operator_code.code = XCOREOpCodes.XC_bconv_int8_DIDO
        else:
            op.operator_code.code = XCOREOpCodes.XC_bconv_bin_DIDO


# Split Bsign operation from Bconv
class SplitBsignPass(LceConv2dPass):
    def match(self, op: Operator) -> bool:

        if not super().match(op):
            return False

        if len(op.inputs) != 2:
            return False

        nobsign = all(
            c.operator_code.code is not XCOREOpCodes.XC_bsign_8
            for i in op.inputs
            for c in i.producers
        )
        return nobsign and op.inputs[0].type is TensorType.INT8

    def mutate(self, op: Operator) -> None:

        subgraph = op.subgraph

        bsign_output = subgraph.create_tensor(
            f"{op.name}/output",
            TensorType.INT32,
            shape=[
                op.inputs[0].shape[0],
                op.inputs[0].shape[1],
                op.inputs[0].shape[2],
                int(op.inputs[0].shape[3] / WORD_SIZE_BITS),
            ],
            consumers=[op],
        )

        bsign_op = subgraph.create_operator(
            OperatorCode(opcode=XCOREOpCodes.XC_bsign_8),
            inputs=[op.inputs[0]],
            outputs=[bsign_output],
        )

        bsign_output.producers = [bsign_op]

        subgraph.insert_operator(op, bsign_op)

        op.inputs = [bsign_op.outputs[0], op.inputs[1]]
        bsign_op.inputs[0].consumers.remove(op)
        bsign_output.buffer.data = np.int32(bsign_output_shape)


# Split out padding to a separate op from BConv
# Note, this currently only matches with BConv but going forward might like to extend this to other conv ops
# and make it a general pass. Bconv only supports SAME and VALID spacial padding
class SplitPaddingFromConvPass(LceConv2dPass):
    def match(self, op: Operator) -> bool:

        if not super().match(op):
            return False

        if len(op.inputs) != 2:
            return False

        return Padding.from_TfLitePadding(op.custom_options["padding"]) is Padding.SAME

    def mutate(self, op: Operator) -> None:

        subgraph = op.subgraph
        options = op.custom_options

        height, width = op.inputs[0].shape[1:3]  # Note, will be in x32 chunks

        C_out, K_h, K_w, C_in = op.inputs[1].shape

        # Cut connection from old input to the op
        old_input = op.inputs[0]
        old_input.consumers.remove(op)

        strides = (options["stride_height"], options["stride_width"])

        # Construct paddings input tensor for PAD op
        padding_tb = int(np.ceil(((strides[0] - 1) * height - strides[0] + K_h) / 2))
        padding_lr = int(np.ceil(((strides[1] - 1) * width - strides[1] + K_w) / 2))
        paddings = [[0, 0], [padding_tb, padding_tb], [padding_lr, padding_lr], [0, 0]]
        padding_tensor = subgraph.create_tensor(
            f"{op.name}/paddings", TensorType.INT32, shape=[4, 2]
        )
        padding_tensor.buffer.data = np.int32(paddings)

        # Note, going foward a builtin.PAD will be inserted, to be replaced by a later pass
        pad_op = subgraph.create_operator(
            OperatorCode(opcode=XCOREOpCodes.XC_pad),
            inputs=[old_input, padding_tensor],
        )

        pad_output_shape = tuple(
            input_dim + sum(pad) for input_dim, pad in zip(old_input.shape, paddings)
        )

        pad_output_tensor = subgraph.create_tensor(
            f"{pad_op.name}/output",
            TensorType.INT32,
            shape=pad_output_shape,
            consumers=[op],
            producers=[pad_op],
        )
        pad_op.outputs = [pad_output_tensor]

        op.inputs[0] = pad_op.outputs[0]

        subgraph.insert_operator(op, pad_op)

        # Pass on pad values from conv to pad op
        pad_op.custom_options["pad_values"] = op.custom_options["pad_values"]

        # Change padding of Bconv from SAME to VALID
        op.custom_options["padding"] = Padding.to_TfLitePadding(Padding.VALID).value
