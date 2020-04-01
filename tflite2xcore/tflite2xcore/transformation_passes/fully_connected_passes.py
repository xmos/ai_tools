# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np

from abc import abstractmethod

from tflite2xcore.operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from tflite2xcore.xcore_model import TensorType
from tflite2xcore.utils import WORD_SIZE
from .transformation_passes import ReplaceXCOREWeightBiasOperatorPass
from .utils import Log


# TODO: write (at least regression) tests for the mutator functions
class ReplaceFullyConnectedPass(ReplaceXCOREWeightBiasOperatorPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.FULLY_CONNECTED

    def mutate_weights(self, op):
        super().mutate_weights(op)
        with self.using(op):
            # zero_padding weight tensor
            col_pad = WORD_SIZE - 1 - (self._weights.shape[1] - 1) % WORD_SIZE
            arr = np.pad(
                self._weights.numpy.astype(np.int8),
                pad_width=[(0, 0),
                           (0, col_pad)]
            )

            # save weight tensor and update shape
            self._weights.buffer.data = arr
            self._weights.shape = arr.shape

            # remove quantization info to save space
            self._weights.quantization = None

    @property
    def _MAX_POST_SHIFT(self):
        return 32 - 16 - 2  # this is because the output is 16 bit

    @Log.output
    def _zero_point_bias(self):
        return np.sum(self._weights.numpy * self._input_zero_point, axis=1)

    def mutate_biases(self, op):
        super().mutate_biases(op)
        with self.using(op):
            # calculate and save the bias/shift/scale tensor
            bss = self._bss_arr()
            self._biases.buffer.data = bss
            self._biases.shape = bss.shape
            self._biases.type = TensorType.INT16

            # rename bias tensor and remove quantization info to save space
            self._biases.name = f"{op.name}/bias_shift_scale"
            self._biases.quantization = None

    def mutate_output(self, op):
        with self.using(op):
            self._output.type = TensorType.INT16
            self._output.name = f"{op.name}/output"
            self._output.quantization = {
                'min': self._output.quantization['min'],
                'max': self._output.quantization['max'],
                'scale': [self._output.quantization['scale'][0] / 2**8],
                'zero_point': [self._output_zero_point * 2**8],
                'details_type': "CustomQuantization",
                'quantized_dimension': 0
            }

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_fc_deepin_anyout)

    @abstractmethod
    def mutate(self, op):
        # NOTE: Overload this in subclasses, and call mutate_output appropriately
        # NOTE: the order of these mutations is strict
        new_op = super().mutate(op)
        self.mutate_biases(new_op)
        self.mutate_weights(new_op)
        return new_op


class ReplaceFullyConnectedOutputPass(ReplaceFullyConnectedPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._output in op.subgraph.outputs

        return False

    def mutate(self, op):
        new_op = super().mutate(op)
        self.mutate_output(new_op)
        return new_op


# TODO: write (at least regression) tests for the mutator functions
class ReplaceFullyConnectedIntermediatePass(ReplaceFullyConnectedPass):
    def match(self, op):
        if super().match(op):
            with self.using(op):
                return self._output not in op.subgraph.outputs

        return False

    def add_requantize(self, op):
        # TODO: this should happen in a separate pass
        with self.using(op):
            # rename original output tensor
            self._output.name = f"{op.name}/output_requant"
            # create intermediate tensor
            intermediate = op.subgraph.create_tensor(
                f"{op.name}/intermediate", self._output.type, self._output.shape,
                quantization=self._output.quantization,
                producers=[op]
            )
        # rewire outputs of original FC op
        for output_tensor in op.outputs:
            output_tensor.producers.remove(op)
        # create new op, insert after original op, rewire inputs/outputs
        new_op = op.subgraph.create_operator(
            OperatorCode(XCOREOpCodes.XC_requantize_16_to_8),
            inputs=[intermediate], outputs=op.outputs)
        op.outputs = [intermediate]
        # move operator to correct location
        # TODO: remove this when we have execution planning
        op.subgraph.insert_operator(op, new_op, after=True)

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        new_op = super().mutate(op)
        self.add_requantize(new_op)
        self.mutate_output(new_op)
        return new_op