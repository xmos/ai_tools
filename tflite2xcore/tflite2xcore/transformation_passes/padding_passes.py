# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy

from tflite2xcore.graph_transformer import OperatorMatchingPass, PassPriority
from tflite2xcore.operator_codes import BuiltinOpCodes, XCOREOpCodes


class FuseConv2dPaddingPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.FUSING):
        super().__init__(priority)

    matching_conv_opcodes = (XCOREOpCodes.XC_conv2d_depthwise,)

    @property
    def _producer(self):
        return self._op.inputs[0].producers[0]

    @property
    def _pad(self):
        return self._op.custom_options['pad']

    @property
    def _pad_params(self):
        return self._producer.inputs[1].numpy.tolist()

    def match(self, op):
        if not super().match(op):
            return False

        with self.using(op):
            opcode = self._op.operator_code.code
            if opcode not in self.matching_conv_opcodes:
                return False

            try:
                pad = self._pad
            except KeyError:
                self.logger.warning(f"{opcode} found without 'pad' option")
                return False

            try:
                if self._producer.operator_code.code is not BuiltinOpCodes.PAD:
                    return False
            except IndexError:
                # No producers found for input
                return False

            pad_params = self._pad_params
            if pad_params[0] != [0, 0] or pad_params[3] != [0, 0]:
                # TODO: a standalone pass should split off channel- and batch-wise padding
                return False

        if len(pad) == 3 and not isinstance(pad, str):
            return True
        elif pad in ['SAME', 'VALID']:
            self.logger.warning(f"Deprecated 'pad' option in {opcode}: 'pad'={pad}")
        else:
            self.logger.warning(f"Invalid option in {opcode}: 'pad'={pad}")

        return False

    def mutate(self, op):
        with self.using(op):
            producer = self._producer
            pad_params = self._pad_params
            old_pad = self._pad
        old_input = op.inputs[0]

        # add connection from unpadded input to convolution operator
        op.inputs[0] = producer.inputs[0]
        producer.inputs[0].consumers.append(op)

        # remove old input and padding op (if it has no other consumers)
        op.subgraph.remove_tensor(old_input)
        if not producer.outputs:
            # NOTE: the paddings tensor might be dangling and will be cleaned up later
            op.subgraph.remove_operator(producer)

        # set padding: [top, left, zero_point]
        op.custom_options['pad'] = [
            old_pad[0] + pad_params[1][0],
            old_pad[1] + pad_params[2][0],
            old_pad[2]
        ]


class SplitPaddingPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.PREP):
        super().__init__(priority)

    @property
    def _pad_params(self):
        return self._op.inputs[1].numpy.tolist()

    def match(self, op):
        if not super().match(op):
            return False

        with self.using(op):
            opcode = self._op.operator_code.code
            if opcode is not BuiltinOpCodes.PAD:
                return False

            pad_params = self._pad_params
            return ((pad_params[0] != [0, 0] or pad_params[3] != [0, 0])
                    and (pad_params[1] != [0, 0] or pad_params[2] != [0, 0]))
