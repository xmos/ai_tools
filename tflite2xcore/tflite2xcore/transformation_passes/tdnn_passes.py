# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
from pprint import pprint
from typing import Iterable

from tflite2xcore.xcore_model import Operator, Subgraph
from tflite2xcore.xcore_schema import BuiltinOpCodes, OperatorCode, XCOREOpCodes

from .conv2d_passes import ReplaceDeepConv2dPass


class TdnnDeepConv2DPass(ReplaceDeepConv2dPass):
    def match(self, op: Operator) -> bool:
        return super().match(op)

    def mutate(self, op: Operator):
        # print(vars(op.inputs[0]))
        # print((op.inputs[1]._shape[1:3]))
        # pprint(vars(op.inputs[1]))

        new_op = super().mutate(op)
        new_op.add_custom_options(tdnn=True)

        pprint(vars(new_op))

        return new_op
