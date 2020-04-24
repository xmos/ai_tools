# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.transformation_passes import QuantizedOperatorMatchingPass
from tflite2xcore.xcore_schema import BuiltinOptions, BuiltinOpCodes, OperatorCode


class CanonicalizeConv2DInputChannels(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    def match(self, op):
        if super().match(op):
            with self.using(op):
                input_shape = self._input.shape
                return len(input_shape) == 4 and input_shape[-1] % 4
        return False

    def mutate(self, op):
        pass
        # subgraph = op.subgraph

        # # create new (batch/channel-wise) operator
        # new_op = subgraph.create_operator(
        #     OperatorCode(BuiltinOpCodes.PAD),
        #     builtin_options_type=BuiltinOptions.PadOptions,
        #     inputs=[],  # TODO:
        # )
        # subgraph.insert_operator(op, new_op)
