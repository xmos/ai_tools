# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.operator_codes import XCOREOpCodes
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    TensorMatchingPass,
    OperatorMatchingPass
)


class RemoveXCOREWeightBiasOperatorQuantInfo(OperatorMatchingPass):
    MATCHING_OP_CODES = [
        XCOREOpCodes.XC_fc_deepin_anyout,
        XCOREOpCodes.XC_conv2d_deep,
        XCOREOpCodes.XC_conv2d_1x1,
        XCOREOpCodes.XC_conv2d_depthwise,
        # TODO: add new shallow conv2d when ready
    ]

    def match(self, op):
        return (super().match(op)
                and op.operator_code.code in self.MATCHING_OP_CODES
                and (op.inputs[1].quantization or op.inputs[2].quantization))

    def mutate(self, op):
        op.inputs[1].quantization = None  # weights
        op.inputs[2].quantization = None  # bss


class RemoveUnusedBuffersPass(ModelTransformationPass):
    # TODO: modify this to use match/mutate
    def run(self, model):
        dangling = [j for j, b in enumerate(model.buffers) if not b.owners]
        if dangling:
            self.logger.debug(f"Found dangling buffers: {dangling}")

        model.buffers = [b for b in model.buffers if b.owners]
        if dangling:
            self.logger.info(f"Removed {len(dangling)} dangling buffers")

        return len(dangling)


class RemoveDanglingTensorsPass(TensorMatchingPass):
    def match(self, tensor):
        return (super().match(tensor)
                and tensor not in tensor.subgraph.inputs
                and tensor not in tensor.subgraph.outputs
                and not tensor.consumers
                and not tensor.producers)

    def mutate(self, tensor):
        tensor.subgraph.remove_tensor(tensor)
