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

        # Note, it is risky to modify an Op in place - create a new op and cut off the old one such that
        # DCE can clean it up
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


# Replace LCEQuantize with XC_BBsign8
class ReplaceLceQuantizePass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:

        try:
            if op.operator_code.code is not ExternalOpCodes.LceQuantize:
                return False
        except AttributeError:
            return False

        return (
            super().match(op)
            # and len(op.inputs) == 1
            # and len(op.outputs) == 1
            and op.outputs[0].type is TensorType.INT32
            and op.inputs[0].type is TensorType.INT8
        )

    def mutate(self, op: Operator) -> None:
        op.operator_code.code = XCOREOpCodes.XC_bsign_8


# Split out padding to a separate op from BConv
# Note, this currently only matches with BConv but going forward might like to extend this to other conv ops
# and make it a general pass. Bconv only supports SAME and VALID spacial padding
class SplitPaddingFromConvPass(LceConv2dPass):
    def match(self, op: Operator) -> bool:

        if not super().match(op):
            return False

        if len(op.inputs) != 3:
            return False

        return op.custom_options["padding"] is Padding.SAME

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
        op.custom_options["padding"] = Padding.VALID


class CanonicalizeLceQuantizedOutputPass(OperatorMatchingPass):
    def match(self, op):
        if super().match(op) and op.operator_code.code is ExternalOpCodes.LceDequantize:
            input_tensor, output_tensor = op.inputs[0], op.outputs[0]
            if (
                output_tensor in op.subgraph.outputs
                and not output_tensor.consumers
                and input_tensor not in op.subgraph.inputs
                and output_tensor.type is TensorType.FLOAT32
                and input_tensor.type is TensorType.INT32
            ):
                if len(output_tensor.producers) == 1:
                    return True
                else:
                    self.logger.warning(
                        "Encountered output of removable LceDequantize "
                        "with more than one producer."
                    )

        return False

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.outputs.append(op.inputs[0])
        subgraph.remove_tensor(op.outputs[0])  # DCE doesn't clean up subgraph outputs
        subgraph.remove_operator(op)
