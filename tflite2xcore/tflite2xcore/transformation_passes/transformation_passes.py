# Copyright (c) 2019, XMOS Ltd, All rights reserved

import logging
import numpy as np

from abc import abstractmethod
from tflite2xcore.graph_transformer import PassPriority
from tflite2xcore.graph_transformer import (
    ModelTransformationPass,
    OperatorMatchingPass,
    InputTensorMatchingPass,
    OutputTensorMatchingPass,
    TensorMatchingPass
)
from tflite2xcore.operator_codes import BuiltinOpCodes, OperatorCode, XCOREOpCodes
from tflite2xcore.xcore_model import TensorType
from tflite2xcore.parallelization import DIDOConv2DPlanner
from tflite2xcore.utils import ACC_PERIOD, WORD_SIZE
from .utils import Log


class RemoveSoftmaxOutputPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.HIGH):
        super().__init__(priority)

    def match(self, op):
        return (super().match(op)
                and op.operator_code.code == BuiltinOpCodes.SOFTMAX
                and op.outputs[0] in op.subgraph.outputs)

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.outputs.append(op.inputs[0])
        subgraph.remove_tensor(op.outputs[0])
        subgraph.remove_operator(op)


class QuantizedOperatorMatchingPass(OperatorMatchingPass):
    def __init__(self, priority=PassPriority.MEDIUM):
        super().__init__(priority)

    @property
    def _output(self):
        return self._op.outputs[0]

    @property
    def _input(self):
        return self._op.inputs[0]

    @property
    def _input_zero_point(self):
        return int(self._input.quantization['zero_point'][0])

    @property
    def _output_zero_point(self):
        return int(self._output.quantization['zero_point'][0])

    @property
    @abstractmethod
    def matching_opcode(self):
        raise NotImplementedError()

    @property
    def _matching_input_type(self):
        return TensorType.INT8

    @property
    def _matching_output_type(self):
        return TensorType.INT8

    def match(self, op):
        if super().match(op) and op.operator_code.code == self.matching_opcode:
            with self.using(op):
                return (self._input.type == self._matching_input_type
                        and self._output.type == self._matching_output_type)


# TODO: write (at least regression) tests for this class
class ReplaceQuantizedOperatorPass(QuantizedOperatorMatchingPass):
    @property
    @abstractmethod
    def new_opcode(self):
        raise NotImplementedError()

    def mutate(self, op):
        new_op = op.subgraph.create_operator(
            self.new_opcode, inputs=op.inputs, outputs=op.outputs)
        new_op.subgraph.replace_operator(op, new_op)
        return new_op


# TODO: write (at least regression) tests for this class
class ReplaceXCOREWeightBiasOperatorPass(ReplaceQuantizedOperatorPass):
    @property
    def _weights(self):
        return self._op.inputs[1]

    def _log_weights(self):
        self.logger.debug(
            "_weights:\n"
            + Log._array_msg(self._weights.numpy.astype(np.int8))
        )

    @property
    def _biases(self):
        return self._op.inputs[2]

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (self._weights.type == TensorType.INT8
                        and self._biases.type == TensorType.INT32)

    def _multiplier(self):
        output_scale = self._output.quantization['scale'][0]
        bias_scale = np.array(self._biases.quantization['scale'])
        return bias_scale / output_scale

    @abstractmethod
    def _zero_point_bias(self):
        pass

    @Log.output
    def _unified_bias(self):
        biases = self._biases.numpy
        return np.int32(biases - self._zero_point_bias()
                        + np.int32(np.round(self._output_zero_point / self._multiplier())))

    @staticmethod
    def __pad_to_acc_period(arr):
        pad = ACC_PERIOD - 1 - (arr.shape[0] - 1) % ACC_PERIOD
        return np.pad(arr, pad_width=[(0, pad)])

    def _bias_arr(self):
        # calculate bias values with the effect of quantization changes
        bias = self._unified_bias()

        # zero pad and reshape
        bias = self.__pad_to_acc_period(bias)
        self.logger.debug("_bias_arr padded biases:\n" + Log._array_msg(bias))

        # splitting lower and upper 16 bits of each 32 bit value
        tmp_shape = (bias.shape[0] // ACC_PERIOD, ACC_PERIOD, -1)
        new_bias = np.frombuffer(bias.flatten().tostring(), dtype=np.int16).reshape(tmp_shape)
        return np.stack([new_bias[:, :, 1], new_bias[:, :, 0]], axis=1)

    def _shift_scale(self):
        multiplier = self._multiplier()
        # NOTE: VLMUL expects one factor in Q2.14
        # we have 1 <= scale < 2 represented in Q2.14
        rshift = -np.ceil(np.log2(multiplier)) + 1
        scale = np.round(2**14 * (multiplier * 2**rshift))

        for j in range(len(scale)):
            if scale[j] == 2**15:
                rshift[j] -= 1
                scale[j] /= 2
            # we are using 16 bits instead of 8 so we need to adjust the shift
            # NOTE: VDEPTH8 shifts down by 8 bits, not 7 as stated on some pages of the ISA
            rshift[j] -= 8

        bias_size = self._biases.numpy.size
        if len(scale) == 1:
            rshift = np.repeat(rshift, bias_size)
            scale = np.repeat(scale, bias_size)
        rshift, scale = np.int16(rshift), np.int16(scale)
        if rshift.shape != scale.shape:
            raise ValueError(f"Shift and scale shapes don't match: {rshift.shape} != {scale.shape}")
        return rshift, scale

    @property
    @abstractmethod
    def _MAX_POST_SHIFT(self):
        pass

    @Log.output
    def _shift_scale_arr(self):
        # calculate right shift/scale
        rshift, scale = self._shift_scale()

        # zero pad and reshape into appropriate array
        new_shape = (-1, ACC_PERIOD)
        rshift = self.__pad_to_acc_period(rshift).reshape(new_shape)
        scale = self.__pad_to_acc_period(scale).reshape(new_shape)

        # split left and right shift into pre and post scaling shifts
        shift_pre = rshift if True else np.maximum(rshift, 0)  # TODO: resolve this when left shift issue is solved in conv2d kernels
        shift_post = self._MAX_POST_SHIFT * np.ones(rshift.shape, dtype=rshift.dtype) + np.minimum(rshift, 0)
        if np.any(shift_post.flatten() < 0):
            raise ValueError("Negative shift_post encountered: "
                             f"{Log._array_msg(shift_post)}")
        return np.stack([shift_pre, scale, shift_post], axis=1)

    def _bss_arr(self):
        # TODO: resolve this when left shift issue is solved in conv2d kernels
        shift_scale_arr = self._shift_scale_arr()
        shift_scale_arr[:, 0, :] = np.maximum(shift_scale_arr[:, 0, :], 0)
        return np.concatenate([self._bias_arr(), shift_scale_arr], axis=1)

    def mutate_biases(self, op):
        # NOTE: by default no bias layout rearrangement is done for this op
        with self.using(op):
            self._biases.name = f"{op.name}/biases"

    def mutate_weights(self, op):
        # NOTE: by default no weight layout rearrangement is done for this op
        with self.using(op):
            self._weights.name = f"{op.name}/weights"


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


class RemoveUnusedBuffersPass(ModelTransformationPass):
    def __init__(self, priority=PassPriority.CLEANUP):
        super().__init__(priority)

    def run(self, model):
        cnt_before = len(model.buffers)
        model.buffers = [b for b in model.buffers if b.owners]
        cnt_removed = cnt_before - len(model.buffers)
        if cnt_removed:
            self.logger.info(f"Removed {cnt_removed} dangling buffers")


class RemoveDanglingTensorsPass(TensorMatchingPass):
    def __init__(self, priority=PassPriority.CLEANUP):
        super().__init__(priority)

    def match(self, tensor):
        return (super().match(tensor)
                and tensor not in tensor.subgraph.inputs
                and tensor not in tensor.subgraph.outputs
                and not tensor.consumers
                and not tensor.producers)

    def mutate(self, tensor):
        tensor.subgraph.remove_tensor(tensor)


class ParallelizeDIDOPass(QuantizedOperatorMatchingPass):
    def __init__(self, priority=PassPriority.PAR, *, num_threads=None, forced=False):
        super().__init__(priority)
        self.num_threads = num_threads or 1
        assert isinstance(self.num_threads, int)
        assert self.num_threads > 0
        self.forced = forced

    def run(self, *args, **kwargs):
        if self.num_threads == 1:
            self.logger.debug(f"Skipping pass b/c num_threads={self.num_threads}")
            return None
        else:
            return super().run(*args, **kwargs)

    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_conv2d_deepin_deepout_relu

    def match(self, op):
        if super().match(op):
            return 'par_plan' not in op.custom_options

    def mutate(self, op):
        with self.using(op):
            _, height, width, _ = self._output.shape
        planner = DIDOConv2DPlanner(
            height, width, num_threads=self.num_threads, forced=self.forced)
        plan = planner.find_optimal_plan()
        op.add_custom_options(par_plan=[list(block) for block in plan.layout])
