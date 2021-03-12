# Copyright 2019-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import logging
import numpy as np
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from tflite2xcore.xcore_schema import TensorType, OperatorCode, Operator, Buffer
from tflite2xcore.utils import ACC_PERIOD_INT8, format_array


class ModelTransformationPass(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _sanity_check(self, obj):
        if __debug__:
            try:
                obj.sanity_check()
            except AssertionError as e:
                self.logger.exception(e)

    @abstractmethod
    def run(self, model):
        return 0

    def __str__(self):
        return self.__class__.__name__


class SubgraphPass(ModelTransformationPass):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._subgraph_idx = -1
        self._obj_index = -1
        self._num_matches = 0

    @abstractmethod
    def match(self, obj) -> bool:
        return True

    @abstractmethod
    def target_iterable(self, subgraph):
        pass

    def log_match(self, obj):
        self.logger.debug(f"matched {obj}")

    @abstractmethod
    def run_subgraph(self, subgraph):
        pass

    def run(self, model):
        modified_cnt = 0
        for self._subgraph_idx, subgraph in enumerate(model.subgraphs):
            self.logger.debug(f"running on subgraph {self._subgraph_idx}")
            if self.run_subgraph(subgraph):
                modified_cnt += 1

        self._subgraph_idx = -1
        return modified_cnt


class SubgraphAnalysisPass(SubgraphPass):
    def run_subgraph(self, subgraph):
        self._num_matches = 0
        for self._obj_index, obj in enumerate(self.target_iterable(subgraph)):
            if self.match(obj):
                self._num_matches += 1
                self.log_match(obj)
                self._sanity_check(obj)
        return 0


class SubgraphTransformationPass(SubgraphPass):
    @abstractmethod
    def mutate(self, obj):
        pass

    def run_subgraph(self, subgraph):
        self._num_matches = 0
        while True:
            for self._obj_index, obj in enumerate(self.target_iterable(subgraph)):
                if self.match(obj):
                    self._num_matches += 1
                    self.log_match(obj)
                    self._sanity_check(obj)
                    self.mutate(obj)
                    self._sanity_check(subgraph)
                    break
            else:
                self._obj_index = -1
                return self._num_matches


class OperatorMatchingPass(SubgraphTransformationPass):
    _op: Operator

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._op = None

    def target_iterable(self, subgraph):
        return subgraph.operators

    @contextmanager
    def using(self, op):
        self._op, original_op = op, self._op
        yield
        self._op = original_op

    def log_match(self, op):
        super().log_match(f"operator [{self._obj_index}]: {op.operator_code}")


class TensorMatchingPass(SubgraphTransformationPass):
    def target_iterable(self, subgraph):
        return subgraph.tensors

    def log_match(self, tensor):
        super().log_match(f"tensor [{self._obj_index}]: {tensor.name}")


class BufferMatchingPass(ModelTransformationPass):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._buffer_idx = -1

    @abstractmethod
    def match(self, buffer):
        return True

    @abstractmethod
    def mutate(self, buffer):
        pass

    def log_match(self, buffer):
        self.logger.debug(
            f"matched buffer [{self._buffer_idx}] of length "
            f"{len(buffer)} with {len(buffer.owners)} owners"
        )

    def run(self, model):
        modified_cnt = 0
        while True:
            for self._buffer_idx, buffer in enumerate(model.buffers):
                if self.match(buffer):
                    self.log_match(buffer)
                    modified_cnt += 1

                    self._sanity_check(buffer)
                    self.mutate(buffer)
                    self._sanity_check(model)
                    break
            else:
                self._buffer_idx = -1
                return modified_cnt


# TODO: add tests
class CanonicalizeEmptyBuffersPass(ModelTransformationPass):
    def run(self, model):
        if model.buffers:
            sentinel = model.buffers[0]
            if not sentinel:  # buffer 0 has to be empty
                for tensor in sentinel.owners:
                    tensor.buffer = Buffer(model)
                    tensor.buffer.owners.append(tensor)
            del model.buffers[0]
            return 1
        return 0


class InputTensorMatchingPass(SubgraphTransformationPass):
    def target_iterable(self, subgraph):
        return subgraph.inputs

    def log_match(self, tensor):
        super().log_match(f"input [{self._obj_index}]: {tensor.name}")


class OutputTensorMatchingPass(SubgraphTransformationPass):
    def target_iterable(self, subgraph):
        return subgraph.outputs

    def log_match(self, tensor):
        super().log_match(f"output [{self._obj_index}]: {tensor.name}")


class QuantizedOperatorMatchingPass(OperatorMatchingPass):
    @property
    def _output(self):
        return self._op.outputs[0]

    @property
    def _input(self):
        return self._op.inputs[0]

    @property
    def _input_zero_point(self):
        return int(self._input.quantization["zero_point"][0])

    @property
    def _output_zero_point(self):
        return int(self._output.quantization["zero_point"][0])

    @property
    @abstractmethod
    def matching_opcode(self):
        raise NotImplementedError()

    @property
    def matching_input_type(self) -> TensorType:
        return TensorType.INT8

    @property
    def matching_output_type(self) -> TensorType:
        return TensorType.INT8

    def match(self, op):
        if super().match(op) and op.operator_code.code is self.matching_opcode:
            with self.using(op):
                return (
                    self._input.type is self.matching_input_type
                    and self._output.type is self.matching_output_type
                )
        return False


class ReplaceQuantizedOperatorPass(QuantizedOperatorMatchingPass):
    @property
    @abstractmethod
    def new_opcode(self) -> OperatorCode:
        raise NotImplementedError()

    def mutate(self, op):
        new_op = op.subgraph.create_operator(
            self.new_opcode, inputs=op.inputs, outputs=op.outputs
        )
        new_op.subgraph.replace_operator(op, new_op)
        return new_op


class ReplaceQuantizedWeightBiasOperatorPass(ReplaceQuantizedOperatorPass):
    @property
    def _weights(self):
        return self._op.inputs[1]

    @property
    def _biases(self):
        return self._op.inputs[2]

    @property
    def matching_biases_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def matching_weights_type(self) -> TensorType:
        return TensorType.INT8

    def _match_non_weight_inputs(self) -> bool:
        try:
            return (
                self._biases.type is self.matching_biases_type
                and self._biases.is_constant
                and self._biases not in self._op.subgraph.outputs
            )
        except IndexError:
            # if bias is missing, the operator should match
            return True

    def match(self, op):
        with self.using(op):
            return (
                super().match(op)
                and self._weights.type is self.matching_weights_type
                # NOTE: the current implementations don't allow mutating ops
                #       if one of the parameter tensors is an output or not constant
                and self._weights.is_constant
                and self._weights not in op.subgraph.outputs
                and self._match_non_weight_inputs()
            )


class ReplaceXCWeightBiasOperatorPass(ReplaceQuantizedWeightBiasOperatorPass):
    def mutate(self, op):
        new_op = super().mutate(op)
        new_op.add_custom_options(illegal_params=True)
        return new_op


# TODO: refactor properties
class LegalizeWeightBiasPass(QuantizedOperatorMatchingPass):
    @property
    def _biases(self):
        return self._op.inputs[2]

    @property
    def _weights(self):
        return self._op.inputs[1]

    @abstractmethod
    def mutate_biases(self, op):
        pass

    @abstractmethod
    def mutate_weights(self, op):
        pass

    def _replace_weights(self, arr) -> None:
        # create and populate new weight tensor
        subgraph = self._op.subgraph
        new_weights = subgraph.create_tensor(
            f"{self._op.name}/weights",
            TensorType.from_numpy_dtype(arr.dtype),
            arr.shape,
            consumers=[self._op],
        )
        new_weights.buffer.data = arr

        # replace old tensor
        self._weights.consumers.remove(self._op)
        self._op.inputs[1] = new_weights

    def match(self, op) -> bool:
        if super().match(op) and "illegal_params" in op.custom_options:
            return op.custom_options["illegal_params"]
        return False

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        self.mutate_biases(op)
        self.mutate_weights(op)
        op.custom_options.pop("illegal_params")
        return op


class LegalizeXCWeightBiasPass(LegalizeWeightBiasPass):
    def _multiplier(self):
        output_scale = self._output.quantization["scale"][0]
        bias_scale = np.array(self._biases.quantization["scale"])
        return bias_scale / output_scale

    @abstractmethod
    def _zero_point_bias(self):
        pass

    def _unified_bias(self):
        arr_64 = self._biases.as_array(np.int64) - self._zero_point_bias().astype(
            np.int64
        )
        arr_32 = np.clip(arr_64, -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
        if np.any(arr_32 != arr_64):
            self.logger.warning("_unified_bias saturated 32 bit!")
        return arr_32

    @staticmethod
    def __pad_to_acc_period(arr):
        pad = ACC_PERIOD_INT8 - 1 - (arr.shape[0] - 1) % ACC_PERIOD_INT8
        return np.pad(arr, pad_width=[(0, pad)])

    def _bias_arr(self):
        # calculate bias values with the effect of quantization changes
        bias = self._unified_bias()

        # zero pad and reshape
        bias = self.__pad_to_acc_period(bias)

        # splitting lower and upper 16 bits of each 32 bit value
        tmp_shape = (bias.shape[0] // ACC_PERIOD_INT8, ACC_PERIOD_INT8, -1)
        new_bias = np.frombuffer(bias.flatten().tobytes(), dtype=np.int16).reshape(
            tmp_shape
        )
        return np.stack([new_bias[:, :, 1], new_bias[:, :, 0]], axis=1)

    @property
    def _SHIFT_ADJUSTMENT(self):
        # NOTE: If we would not need to add the offset separately, the intermediate
        #       could never saturate, and this value would be 8. But decreasing to 7
        #       means that we get an extra bit of headroom in the intermediate.
        # TODO: investigate if this could be calculated/estimated from the parameters
        return 7

    def _shift_scale(self):
        multiplier = self._multiplier()
        # NOTE: VLMUL expects one factor in Q2.14
        # we have 1 <= scale < 2 represented in Q2.14

        multiplier_mask = multiplier != 0
        rshift = np.full(multiplier.shape, 16)
        rshift[multiplier_mask] = -np.ceil(np.log2(multiplier[multiplier_mask])) + 1
        scale = np.full(multiplier.shape, 2 ** 15 - 1)
        scale[multiplier_mask] = np.round(
            multiplier[multiplier_mask] * 2 ** (14 + rshift[multiplier_mask])
        )

        for j in range(len(scale)):
            if scale[j] == 2 ** 15:
                rshift[j] -= 1
                scale[j] /= 2
            # we are using 16 bits instead of 8 so we need to adjust the shift
            rshift[j] -= self._SHIFT_ADJUSTMENT

        bias_size = np.prod(self._biases.shape)
        if len(scale) == 1:
            rshift = np.repeat(rshift, bias_size)
            scale = np.repeat(scale, bias_size)
        rshift, scale = np.int16(rshift), np.int16(scale)
        if rshift.shape != scale.shape:
            raise ValueError(
                f"Shift and scale shapes don't match: {rshift.shape} != {scale.shape}"
            )
        return rshift, scale

    @property
    def _OUTPUT_BITS(self):
        return 8

    @property
    def _MAX_POST_SHIFT(self):
        return 22 + self._SHIFT_ADJUSTMENT - self._OUTPUT_BITS

    def _scale_offset_arr(self):
        # calculate right shift/scale
        rshift, scale = self._shift_scale()

        # zero pad and reshape into appropriate array
        rshift = self.__pad_to_acc_period(rshift)
        scale = self.__pad_to_acc_period(scale)

        # split left and right shift into pre and post scaling shifts
        shift_pre = np.maximum(rshift, 0)
        shift_post = self._MAX_POST_SHIFT * np.ones(
            rshift.shape, dtype=rshift.dtype
        ) + np.minimum(rshift, 0)
        if np.any(shift_post.flatten() < 0):
            raise ValueError(
                "Negative shift_post encountered: " f"{format_array(shift_post)}"
            )

        # calculate offset
        raw_offset = (
            np.float64(self._output_zero_point)
            * 2 ** shift_post.astype(np.float64)
            * 2 ** (self._OUTPUT_BITS - 8)
        ).flatten()

        offset_scale = np.round(np.sqrt(np.abs(raw_offset))).astype(np.int16)
        offset = np.zeros(offset_scale.shape, dtype=offset_scale.dtype)
        pos_ind = offset_scale > 0
        offset[pos_ind] = np.round(raw_offset[pos_ind] / offset_scale[pos_ind]).astype(
            np.int16
        )

        new_shape = (-1, ACC_PERIOD_INT8)
        return np.stack(
            [
                shift_pre.reshape(new_shape),
                scale.reshape(new_shape),
                offset_scale.reshape(new_shape),
                offset.reshape(new_shape),
                shift_post.reshape(new_shape),
            ],
            axis=1,
        )

    def _bso_arr(self):
        return np.concatenate([self._bias_arr(), self._scale_offset_arr()], axis=1)

    def __add_const_zero_bias(self):
        out_channels = self._output.shape[-1]
        input_scale = self._input.quantization["scale"][0]
        new_biases = self._op.subgraph.create_tensor(
            f"{self._op.name}/const_zero_bias",
            TensorType.INT32,
            shape=(out_channels,),
            consumers=[self._op],
            quantization={
                "scale": [
                    input_scale * weight_scale
                    for weight_scale in self._weights.quantization["scale"]
                ],
                "zero_point": [0] * out_channels,
            },
        )
        new_biases.buffer.data = np.zeros(new_biases.shape, dtype=np.int32)
        self._op.inputs.append(new_biases)

    def mutate_biases(self, op):
        with self.using(op):
            try:
                self._biases
            except IndexError:
                self.__add_const_zero_bias()

            # calculate the bias/scale/offset tensor
            bso = self._bso_arr()

            # create and populate new bias tensor
            new_biases = self._op.subgraph.create_tensor(
                f"{self._op.name}/bias_shift_scale",
                TensorType.INT16,
                bso.shape,
                consumers=[self._op],
            )
            new_biases.buffer.data = bso

            # replace old tensor
            self._biases.consumers.remove(self._op)
            self._op.inputs[2] = new_biases
