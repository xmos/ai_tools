# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np

from copy import deepcopy

from typing import overload

from tflite2xcore.utils import WORD_SIZE
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
    CustomOpCode,
    XCOREOpCodes,
    OperatorCode,
    BuiltinOptions,
)

def SupportedBconv2DOp(op: Operator) -> bool:

    # isinstance - yuk!
    if not (
        isinstance(op.operator_code.code, CustomOpCode)
        and op.operator_code.custom_code.name == "LceBconv2d"
    ):
        return False

    options = op.custom_options

    try:
        strides = (options["stride_height"], options["stride_width"])
        dilations = (
            options["dilation_height_factor"],
            options["dilation_width_factor"],
        )
        weights = op.inputs[1]
    except KeyError:
        return False

    # TODO check padding
    return (
        strides == (1, 1)
        and dilations == (1, 1)
        and weights.shape[0] % 32 == 0  # Cout
        and (weights.shape[3] * 32) % 256 == 0 # Cin
        and weights.type == TensorType.INT32
        and (
            op.inputs[0].type == TensorType.INT8
            or op.inputs[0].type == TensorType.INT32
        )
    )


class CanonicalizeLceBconv2DPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        return SupportedBconv2DOp(op) and len(op.inputs) == 4

    def mutate(self, op: Operator) -> None:
        op.inputs[2].consumers.remove(op)
        op.inputs[3].consumers.remove(op)
        op.inputs = op.inputs[:2]

# Replace LCEBconv2D with XC_BConv2D
class ReplaceLceBconv2DPass(OperatorMatchingPass):
    def __init__(
        self, input_tensor_type: TensorType, *args: str, **kwargs: int
    ) -> None:
        super().__init__(*args, **kwargs)
        self._input_tensor_type = input_tensor_type

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and SupportedBconv2DOp(op)
            and len(op.inputs) == 2
            and op.inputs[0].type == self._input_tensor_type
        )

    def mutate(self, op: Operator) -> None:
        if self._input_tensor_type == TensorType.INT8:
            op.operator_code.custom_code = XCOREOpCodes.XC_bconv_int8_DIDO
        else:
            op.operator_code.custom_code = XCOREOpCodes.XC_bconv_bin_DIDO


# Split Bsign operation from Bconv
class SplitBsignPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:

        if not super().match(op):
            return False

        match = SupportedBconv2DOp(op) and len(op.inputs) == 2

        nobsign = all(
            all(
                (c.operator_code.code is not XCOREOpCodes.XC_bsign_8)
                for c in i.producers
            )
            for i in op.inputs
        )
        return match and nobsign and op.inputs[0].type == TensorType.INT8

    def mutate(self, op: Operator) -> None:

        subgraph = op.subgraph

        bsign_op = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.CUSTOM, custom_code=XCOREOpCodes.XC_bsign_8),
            inputs=[op.inputs[0]],
        )

        subgraph.insert_operator(op, bsign_op)

        bsign_op.outputs.append(
            subgraph.create_tensor(
                f"{op.name}/output",
                TensorType.INT32,
                shape=[
                    op.inputs[0].shape[0],
                    int(op.inputs[0].shape[1] / 32),
                    int(op.inputs[0].shape[2] / 32),
                    op.inputs[0].shape[3],
                ],
                producers=[bsign_op],
                consumers=[op],
            )
        )

        op.inputs = [bsign_op.outputs[0], op.inputs[1]]
        bsign_op.inputs[0].consumers.remove(op)


# Split out padding to a seperate of from BConv
# Note, this currently only matches with BConv but going forward might like to extend this to other conv ops
# and make it a general pass
class SplitPaddingFromConvPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:

        if not super().match(op):
            return False

        if not (SupportedBconv2DOp(op) and len(op.inputs) == 2):
            return False

        return op.custom_options["padding"] == 1  # kTfLitePaddingSame

    def mutate(self, op: Operator) -> None:

        subgraph = op.subgraph

        op.custom_options["padding"] = 2  # kTfLitePaddingValid

        # Cut connection from old input to the op
        old_input = op.inputs[0]
        old_input.consumers.remove(op)

        # TODO
        pads = [[0, 0], [0, 0], [0, 0], [0, 0]]

        padding_tensor = subgraph.create_tensor(
            f"{op.name}/paddings", TensorType.INT32, shape=[4, 2]
        )

        # Insert PAD op
        pad_op = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.PAD), inputs=[old_input, padding_tensor],
        )

        pad_output_tensor = subgraph.create_tensor(
            f"{pad_op.name}/output",
            TensorType.INT32,
            shape=old_input.shape,
            consumers=[op],
            producers=[pad_op],
        )  # TODO fix shape

        pad_op.outputs = [pad_output_tensor]

        op.inputs[0] = pad_op.outputs[0]

        subgraph.insert_operator(op, pad_op)

        pad_op.inputs[1].buffer.data = np.int32(pads)

        pad_values = op.custom_options["pad_values"]
