# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np
from copy import deepcopy

from tflite2xcore.xcore_schema import TensorType, BuiltinOpCodes, OperatorCode

from .transformation_passes import QuantizedOperatorMatchingPass


class CanonicalizeConv2DInputChannels(QuantizedOperatorMatchingPass):
    @property
    def _weights(self):
        return self._op.inputs[1]

    @property
    def _biases(self):
        return self._op.inputs[2]

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
        subgraph = op.subgraph

        with self.using(op):
            old_weight_tensor = self._weights
            old_shape = old_weight_tensor.shape
            new_shape = [*old_shape[:3], int(4 * np.ceil(old_shape[3] / 4))]
            pad_size = new_shape[3] - old_shape[3]

            pads = [[0, 0], [0, 0], [0, 0], [0, pad_size]]

            # create new zero padded kernel tensor
            new_weight_tensor = subgraph.create_tensor(
                f"{self._op.name}/weights",
                old_weight_tensor.type,
                new_shape,
                isinput=old_weight_tensor in subgraph.inputs,
                isoutput=old_weight_tensor in subgraph.outputs,
                quantization=old_weight_tensor.quantization,
                consumers=[self._op],
            )
            new_weight_tensor.buffer.data = np.pad(self._weights.as_array(), pads)

            # rewire old and new kernel tensors
            old_weight_tensor.consumers.remove(self._op)
            self._op.inputs[1] = new_weight_tensor

            # create new channel-wise padding operator
            old_input = self._input
            pad_op = subgraph.create_operator(
                OperatorCode(BuiltinOpCodes.PAD), inputs=[old_input],
            )
            subgraph.insert_operator(self._op, pad_op)
            old_input.consumers.remove(self._op)

            # create paddings tensor and connect to op
            paddings_tensor = subgraph.create_tensor(
                f"{pad_op.name}/paddings",
                TensorType.INT32,
                shape=[4, 2],
                consumers=[pad_op],
            )
            paddings_tensor.buffer.data = np.int32(pads)
            pad_op.inputs.append(paddings_tensor)

            # create intermediate tensor and wire up to conv and pad ops
            self._op.inputs[0] = subgraph.create_tensor(
                f"{self._op.name}/xc_padded_input",
                TensorType.INT8,
                shape=[*old_input.shape[:3], new_shape[3]],
                producers=[pad_op],
                consumers=[self._op],
                quantization=deepcopy(old_input.quantization),
            )
            pad_op.outputs.append(self._op.inputs[0])
