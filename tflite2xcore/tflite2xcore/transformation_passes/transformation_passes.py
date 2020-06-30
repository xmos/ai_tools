# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy as np

from abc import abstractmethod
from contextlib import contextmanager

from tflite2xcore.pass_manager import ModelTransformationPass
from tflite2xcore.xcore_schema import TensorType, BuiltinOpCodes
from tflite2xcore.utils import ACC_PERIOD
from tflite2xcore import xlogging as logging


class SubgraphTransformationPass(ModelTransformationPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subgraph_idx = -1
        self._obj_index = -1

    @abstractmethod
    def match(self, obj):
        return True

    @abstractmethod
    def mutate(self, obj):
        pass

    @abstractmethod
    def target_iterable(self, subgraph):
        pass

    def log_match(self, obj):
        self.logger.info(f"matched {obj}")

    def run_subgraph(self, subgraph):
        num_matches = 0
        while True:
            for self._obj_index, obj in enumerate(self.target_iterable(subgraph)):
                if self.match(obj):
                    num_matches += 1
                    self.log_match(obj)

                    if self.debug:
                        try:
                            obj.sanity_check()
                        except AssertionError as e:
                            self.logger.exception(e)
                        import pdb

                        pdb.set_trace()

                    self.mutate(obj)
                    break
            else:
                self._obj_index = -1
                return num_matches

    def run(self, model):
        modified_cnt = 0
        for self._subgraph_idx, subgraph in enumerate(model.subgraphs):
            self.logger.debug(f"running on subgraph {self._subgraph_idx}")
            if self.run_subgraph(subgraph):
                modified_cnt += 1

            if self.debug:
                try:
                    subgraph.sanity_check()
                except AssertionError as e:
                    self.logger.exception(e)

        self._subgraph_idx = -1
        return modified_cnt


class OperatorMatchingPass(SubgraphTransformationPass):
    def __init__(self, *args, **kwargs):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer_idx = -1

    @abstractmethod
    def match(self, buffer):
        return True

    @abstractmethod
    def mutate(self, buffer):
        pass

    def log_match(self, buffer):
        self.logger.info(
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

                    if self.debug:
                        try:
                            model.sanity_check()
                        except AssertionError as e:
                            self.logger.exception(e)
                        import pdb

                        pdb.set_trace()

                    self.mutate(buffer)
                    break
            else:
                self._buffer_idx = -1
                return modified_cnt


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


class RemoveSoftmaxOutputPass(OperatorMatchingPass):
    def match(self, op):
        return (
            super().match(op)
            and op.operator_code.code == BuiltinOpCodes.SOFTMAX
            and op.outputs[0] in op.subgraph.outputs
        )

    def mutate(self, op):
        subgraph = op.subgraph
        subgraph.outputs.append(op.inputs[0])
        subgraph.remove_tensor(op.outputs[0])
        subgraph.remove_operator(op)


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
    def _matching_input_type(self):
        return TensorType.INT8

    @property
    def _matching_output_type(self):
        return TensorType.INT8

    def match(self, op):
        if super().match(op) and op.operator_code.code == self.matching_opcode:
            with self.using(op):
                return (
                    self._input.type == self._matching_input_type
                    and self._output.type == self._matching_output_type
                )


# TODO: write (at least regression) tests for this class
class ReplaceQuantizedOperatorPass(QuantizedOperatorMatchingPass):
    @property
    @abstractmethod
    def new_opcode(self):
        raise NotImplementedError()

    def mutate(self, op):
        new_op = op.subgraph.create_operator(
            self.new_opcode, inputs=op.inputs, outputs=op.outputs
        )
        new_op.subgraph.replace_operator(op, new_op)
        return new_op


class ReplaceWeightBiasOperatorPass(ReplaceQuantizedOperatorPass):
    @property
    def _weights(self):
        return self._op.inputs[1]

    def _log_weights(self):
        self.logger.xdebug(
            "_weights:\n" + logging._array_msg(self._weights.numpy.astype(np.int8))
        )

    @property
    def _biases(self):
        return self._op.inputs[2]

    def match(self, op):
        if super().match(op):
            with self.using(op):
                return (
                    self._weights.type is TensorType.INT8
                    and self._biases.type is TensorType.INT32
                )

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

    def _log_weights(self):
        self.logger.xdebug(
            "_weights:\n" + logging._array_msg(self._weights.numpy.astype(np.int8))
        )

    @abstractmethod
    def mutate_biases(self, op):
        pass

    @abstractmethod
    def mutate_weights(self, op):
        pass

    def _replace_weights(self, arr):
        # create and populate new weight tensor
        subgraph = self._op.subgraph
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

    def match(self, op):
        if super().match(op) and "illegal_params" in op.custom_options:
            return op.custom_options["illegal_params"]

    def mutate(self, op):
        # NOTE: the order of these mutations is strict
        self.mutate_biases(op)
        self.mutate_weights(op)
        op.custom_options.pop("illegal_params")


class LegalizeXCWeightBiasPass(LegalizeWeightBiasPass):
    def _multiplier(self):
        output_scale = self._output.quantization["scale"][0]
        bias_scale = np.array(self._biases.quantization["scale"])
        return bias_scale / output_scale

    @abstractmethod
    def _zero_point_bias(self):
        pass

    @logging.log_method_output()
    def _unified_bias(self):
        biases = self._biases.numpy
        arr_64 = biases.astype(np.int64) - self._zero_point_bias().astype(np.int64)
        arr_32 = np.clip(arr_64, -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
        if np.any(arr_32 != arr_64):
            self.logger.warning("_unified_bias saturated 32 bit!")
        return arr_32

    @staticmethod
    def __pad_to_acc_period(arr):
        pad = ACC_PERIOD - 1 - (arr.shape[0] - 1) % ACC_PERIOD
        return np.pad(arr, pad_width=[(0, pad)])

    def _bias_arr(self):
        # calculate bias values with the effect of quantization changes
        bias = self._unified_bias()

        # zero pad and reshape
        bias = self.__pad_to_acc_period(bias)
        self.logger.xdebug("_bias_arr padded biases:\n" + logging._array_msg(bias))

        # splitting lower and upper 16 bits of each 32 bit value
        tmp_shape = (bias.shape[0] // ACC_PERIOD, ACC_PERIOD, -1)
        new_bias = np.frombuffer(bias.flatten().tostring(), dtype=np.int16).reshape(
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
        rshift = -np.ceil(np.log2(multiplier)) + 1
        scale = np.round(2 ** 14 * (multiplier * 2 ** rshift))

        for j in range(len(scale)):
            if scale[j] == 2 ** 15:
                rshift[j] -= 1
                scale[j] /= 2
            # we are using 16 bits instead of 8 so we need to adjust the shift
            rshift[j] -= self._SHIFT_ADJUSTMENT

        bias_size = self._biases.numpy.size
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
    @abstractmethod
    def _OUTPUT_BITS(self):
        pass

    @property
    def _MAX_POST_SHIFT(self):
        return 22 + self._SHIFT_ADJUSTMENT - self._OUTPUT_BITS

    @logging.log_method_output()
    def _scale_offset_arr(self):
        # calculate right shift/scale
        rshift, scale = self._shift_scale()

        # zero pad and reshape into appropriate array
        new_shape = (-1, ACC_PERIOD)
        rshift = self.__pad_to_acc_period(rshift)
        scale = self.__pad_to_acc_period(scale)

        # split left and right shift into pre and post scaling shifts
        shift_pre = np.maximum(rshift, 0)
        shift_post = self._MAX_POST_SHIFT * np.ones(
            rshift.shape, dtype=rshift.dtype
        ) + np.minimum(rshift, 0)
        if np.any(shift_post.flatten() < 0):
            raise ValueError(
                "Negative shift_post encountered: " f"{logging._array_msg(shift_post)}"
            )

        # calculate offset
        raw_offset = (
            np.float64(self._output_zero_point)
            * 2 ** shift_post.astype(np.float64)
            * 2 ** (self._OUTPUT_BITS - 8)
        ).flatten()

        self.logger.xdebug(
            "raw_offset:\n" + logging._array_msg(raw_offset.astype(np.int32))
        )
        offset_scale = np.round(np.sqrt(np.abs(raw_offset))).astype(np.int16)
        offset = np.zeros(offset_scale.shape, dtype=offset_scale.dtype)
        pos_ind = offset_scale > 0
        offset[pos_ind] = np.round(raw_offset[pos_ind] / offset_scale[pos_ind]).astype(
            np.int16
        )

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

    def mutate_biases(self, op):
        with self.using(op):
            subgraph = self._op.subgraph

            # calculate the bias/scale/offset tensor
            bso = self._bso_arr()

            # create and populate new bias tensor
            new_biases = subgraph.create_tensor(
                f"{self._op.name}/bias_shift_scale",
                TensorType.INT16,
                bso.shape,
                isinput=self._biases in subgraph.inputs,
                isoutput=self._biases in subgraph.outputs,
                consumers=[self._op],
            )
            new_biases.buffer.data = bso

            # replace old tensor
            self._biases.consumers.remove(self._op)
            self._op.inputs[2] = new_biases


class RemovePaddingInputPass(OperatorMatchingPass):
    def match(self, op):
       
        #Match operator
        if op.operator_code.code is BuiltinOpCodes.PAD:
            padding = op.inputs[1].numpy.tolist()
 
            return (
                super().match(op)
                # Match positon in subgraph
                and op.inputs[0] in op.subgraph.inputs
                #Match only padding in channel direction i.e. inserted for VPU alignment
                and len(padding) == 4
                and (padding[-1] != [0,0])
                and all(pad == [0,0] for pad in padding[:-1])
            )
            
        else:
            return False

    def mutate(self, op):
        subgraph = op.subgraph

        subgraph.inputs.append(op.outputs[0])
        subgraph.remove_tensor(op.inputs[0])
        subgraph.remove_operator(op)


