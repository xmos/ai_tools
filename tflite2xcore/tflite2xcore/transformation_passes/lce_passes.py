# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np

from copy import deepcopy

from tflite2xcore.xcore_schema import (
    BuiltinOpCodes,
    OperatorCode,
    XCOREOpCodes,
    CustomOpCode,
)
from tflite2xcore.utils import WORD_SIZE
from .transformation_passes import (
    OperatorMatchingPass,
    LegalizeWeightBiasPass,
    LegalizeXCWeightBiasPass,
)
from tflite2xcore.xlogging import log_method_output
from tflite2xcore.xcore_model import Operator, Tensor

from tflite2xcore.transformation_passes import OperatorMatchingPass
from tflite2xcore.xcore_schema import (
    Padding,
    TensorType,
    BuiltinOpCodes,
    XCOREOpCodes,
    OperatorCode,
    BuiltinOptions,
)

def SupportedBconv2DOp(op: Operator) -> bool:

    # isinstance - yuk!
    if not (isinstance(op.operator_code.code, CustomOpCode) and op.operator_code.custom_code.name == "LceBconv2d"):
        return False

    options = op.builtin_options
    strides = (options["stride_h"], options["stride_w"])
    dilations = (options["dilation_h_factor"], options["dilation_w_factor"])
    weights = op.inputs[1]

    return (
        strides == (1, 1)
        and dilations == (1, 1)
        and weights.shape[0] % 32 == 0
        and weights.shape[3] % 256 == 0 
        and weights.type == TensorType.INT32
        and (op.inputs[0].type == TensorType.INT8 or op.inputs[0].type == TensorType.INT32)
    )

class CanonicalizeLceBconv2DPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        return SupportedBconv2DOp(op) and len(op.inputs) == 4

    def mutate(self, op: Operator) -> None:
        op.inputs[2].consumers.remove(op)
        op.inputs[3].consumers.remove(op)
        op.inputs = op.inputs[:2]


class ReplaceLceBconv2DPass(OperatorMatchingPass):
    def match(self, op):

        if not super().match(op):
            return False

        return (
            SupportedBconv2DOp(op)
            and len(op.inputs) == 2
        )


class ReplaceLceBconv2DPass_int8(ReplaceLceBconv2DPass):
    def match(self, op):
        return super().match(op) and op.inputs[0].type == TensorType.INT8

    def mutate(self, op):
        op.operator_code.custom_code = XCOREOpCodes.XC_bconv_deep


class ReplaceLceBconv2DPass_bitpacked(ReplaceLceBconv2DPass):
    def match(self, op):
        return super().match(op) and op.inputs[0].type == TensorType.INT32

    def mutate(self, op):
        op.operator_code.custom_code = XCOREOpCodes.XC_bconv_deep


class InsertBsignPass(OperatorMatchingPass):
    def match(self, op):

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

    def mutate(self, op):
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
