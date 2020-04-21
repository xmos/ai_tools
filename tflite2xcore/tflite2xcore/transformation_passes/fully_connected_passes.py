# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np

from tflite2xcore.operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from tflite2xcore.xcore_model import TensorType
from tflite2xcore.utils import WORD_SIZE
from .transformation_passes import (
    ReplaceWeightBiasOperatorPass,
    QuantizedOperatorMatchingPass,
    LegalizeXCBiasPass,
)
from tflite2xcore.xlogging import log_method_output


# TODO: write (at least regression) tests for the mutator functions
class ReplaceFullyConnectedPass(ReplaceWeightBiasOperatorPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.FULLY_CONNECTED

    def mutate_output(self, op):
        # TODO: revise this when FC becomes 8bit output
        with self.using(op):
            self._output.type = TensorType.INT16
            new_quantization = {
                k: v
                for k, v in self._output.quantization.items()
                if k in ["min", "max"]
            }
            new_quantization.update(
                {
                    "scale": [self._output.quantization["scale"][0] / 2 ** 8],
                    "zero_point": [self._output_zero_point * 2 ** 8],
                    "details_type": "CustomQuantization",
                    "quantized_dimension": 0,
                }
            )
            self._output.quantization = new_quantization

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_fc_deepin_anyout)

    def add_requantize(self, op):
        with self.using(op):
            # create intermediate tensor
            intermediate = op.subgraph.create_tensor(
                f"{op.name}/intermediate",
                self._output.type,
                self._output.shape,
                quantization=self._output.quantization,
                producers=[op],
            )
        # rewire outputs of original FC op
        for output_tensor in op.outputs:
            output_tensor.producers.remove(op)
        # create new op, insert after original op, rewire inputs/outputs
        new_op = op.subgraph.create_operator(
            OperatorCode(XCOREOpCodes.XC_requantize_16_to_8),
            inputs=[intermediate],
            outputs=op.outputs,
        )
        op.outputs = [intermediate]
        # move operator to correct location
        # TODO: remove this when we have execution planning
        op.subgraph.insert_operator(op, new_op, after=True)

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        new_op = super().mutate(op)
        self.add_requantize(new_op)
        self.mutate_output(new_op)
        new_op.add_custom_options(illegal_inputs=[1, 2])
        return new_op


class LegalizeXCFullyConnectedWeightPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_fc_deepin_anyout

    @property
    def _matching_output_type(self):
        return TensorType.INT16

    @property
    def _weights(self):
        return self._op.inputs[1]

    def match(self, op):
        if super().match(op) and "illegal_inputs" in op.custom_options:
            return 1 in op.custom_options["illegal_inputs"]

    def mutate(self, op):
        with self.using(op):
            subgraph = self._op.subgraph

            # zero_padding weight tensor
            col_pad = WORD_SIZE - 1 - (self._weights.shape[1] - 1) % WORD_SIZE
            arr = np.pad(
                self._weights.numpy.astype(np.int8), pad_width=[(0, 0), (0, col_pad)]
            )

            # create and populate new weight tensor
            new_weights = subgraph.create_tensor(
                f"{self._op.name}/weights",
                TensorType.INT8,
                arr.shape,
                isinput=self._weights in subgraph.inputs,
                isoutput=self._weights in subgraph.outputs,
                consumers=[self._op],
            )
            new_weights.buffer.data = arr

            # replace old tensor
            self._weights.consumers.remove(self._op)
            self._op.inputs[1] = new_weights

            self._op.custom_options["illegal_inputs"].remove(1)


class LegalizeXCFullyConnectedBiasPass(LegalizeXCBiasPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_fc_deepin_anyout

    @property
    def _matching_output_type(self):
        return TensorType.INT16

    @property
    def _MAX_POST_SHIFT(self):
        return 32 - 16 - 2  # this is because the output is 16 bit

    @log_method_output()
    def _zero_point_bias(self):
        return np.sum(self._weights.numpy * self._input_zero_point, axis=1)

    def match(self, op):
        if super().match(op) and "illegal_inputs" in op.custom_options:
            return 2 in op.custom_options["illegal_inputs"]
