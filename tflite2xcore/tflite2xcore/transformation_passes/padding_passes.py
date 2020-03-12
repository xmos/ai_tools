# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging

from tflite2xcore.graph_transformer import OperatorMatchingPass, PassPriority
from tflite2xcore.operator_codes import BuiltinOpCodes, XCOREOpCodes


class FuseConv2dPaddingPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.FUSING):
        super().__init__(priority)

    matching_conv_opcodes = (XCOREOpCodes.XC_conv2d_depthwise,)

    def match(self, op):
        if not super().match(op):
            return False

        opcode = op.operator_code.code
        if opcode not in self.matching_conv_opcodes:
            return False

        try:
            padding = op.custom_options['pad']
        except KeyError:
            logging.warning(f"{opcode} found without 'pad' option")
            return False

        try:
            producer = op.inputs[0].producers[0]
        except IndexError:
            # No producers found for input
            return False
        else:
            if producer.operator_code.code is not BuiltinOpCodes.PAD:
                return False

        pad_params = producer.inputs[1].numpy.tolist()
        if pad_params[0] != [0, 0] or pad_params[3] != [0, 0]:
            # TODO: a standalone pass should split off channel- and batch-wise padding
            return False

        if padding is not 'VALID':
            # TODO: implement fusing for at least some of these cases (e.g. 'SAME')
            logging.warning("Cannot fuse PAD operator followed by "
                            f"{opcode} with padding {padding}")
            return False

        return True

    def mutate(self, op):
        raise NotImplementedError()
