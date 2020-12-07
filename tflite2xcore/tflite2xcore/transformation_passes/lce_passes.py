# Copyright (c) 2020, XMOS Ltd, All rights reserved
import numpy as np
from math import ceil
from typing import Tuple, List

from tflite2xcore.utils import (
    WORD_SIZE_BITS,
    VECTOR_SIZE_BITS,
    VECTOR_SIZE_WORDS,
    ACC_PERIOD,
    WORD_SIZE,
    xor_popcount,
    calculate_same_padding,
    get_unpacked_shape,
)
from tflite2xcore.xcore_schema import (
    Operator,
    Padding,
    TensorType,
    ExternalOpCodes,
    XCOREOpCodes,
    OperatorCode,
    BuiltinOpCodes,
)

from .transformation_passes import (
    OperatorMatchingPass,
    ReplaceQuantizedOperatorPass,
    LegalizeWeightBiasPass,
)
from .conv2d_passes import ReplaceConv2DPass

FILLER = 0x55555555

XC_BCONV2D_OPCODES = (
    XCOREOpCodes.XC_bconv2d_bin,
    XCOREOpCodes.XC_bconv2d_bin_DI,
    XCOREOpCodes.XC_bconv2d_int8,
    XCOREOpCodes.XC_bconv2d_int8_DIDO,
)


class ReplaceBconv2DPass(ReplaceConv2DPass):
    @property
    def matching_opcode(self) -> ExternalOpCodes:
        return ExternalOpCodes.LceBconv2d

    @property
    def matching_input_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def matching_biases_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def matching_weights_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def _strides(self) -> Tuple[int, int]:
        options = self._op.custom_options
        return options["stride_height"], options["stride_width"]

    @property
    def _dilation(self) -> Tuple[int, int]:
        options = self._op.custom_options
        return options["dilation_height_factor"], options["dilation_width_factor"]

    @property
    def _padding(self) -> Padding:
        return self._op.custom_options["padding"]

    @property
    def _input_channels(self) -> int:
        return self._op.custom_options["channels_in"]

    @property
    def _output_channels(self) -> int:
        return self._weights.shape[0]

    def match(self, op: Operator) -> bool:
        if super().match(op):
            with self.using(op):
                if self._input_channels != self._weights.shape[3] * WORD_SIZE_BITS:
                    self.logger.warning(
                        f"Found {self.matching_opcode} operator "
                        f"with {self._input_channels} input channels "
                        f"(not a multiple of {WORD_SIZE_BITS})."
                    )
                elif self._output_channels % WORD_SIZE != 0:
                    self.logger.warning(
                        f"Found {self.matching_opcode} operator "
                        f"with {self._output_channels} output channels "
                        f"(not a multiple of {WORD_SIZE})"
                    )
                else:
                    return True

        return False

    def mutate(self, op: Operator) -> None:
        new_op = super().mutate(op)
        with self.using(op):
            new_op.add_custom_options(
                stride=self._strides, padding=self._padding,
            )
        return new_op


class ReplaceBconv2DInt8Pass(ReplaceBconv2DPass):
    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_bconv2d_int8)

    def _match_non_weight_inputs(self) -> bool:
        return len(self._op.inputs) == 4 and all(
            params_tensor.type is TensorType.FLOAT32
            and params_tensor.is_constant
            and params_tensor not in self._op.subgraph.outputs
            for params_tensor in self._op.inputs[2:]
        )


class ReplaceBconv2DInt8DeepInDeepOutPass(ReplaceBconv2DInt8Pass):
    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_bconv2d_int8_DIDO)

    def match(self, op: Operator) -> bool:
        with self.using(op):
            return (
                super().match(op)
                and self._input_channels % VECTOR_SIZE_BITS == 0
                and self._output_channels % ACC_PERIOD == 0
            )


class ReplaceBconv2DBitpackedPass(ReplaceBconv2DPass):
    @property
    def matching_output_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_bconv2d_bin)

    def _match_non_weight_inputs(self) -> bool:
        return (
            len(self._op.inputs) == 3
            and self._op.inputs[2].type is TensorType.INT32
            and self._op.inputs[2].is_constant
            and self._op.inputs[2] not in self._op.subgraph.outputs
        )

    def match(self, op: Operator) -> bool:
        if super().match(op):
            with self.using(op):
                if self._output_channels % WORD_SIZE_BITS == 0:
                    return True
                self.logger.warning(
                    f"Found {self.matching_opcode} operator with bitpacked output "
                    f"and {self._output_channels} output channels "
                    f"(not a multiple of {WORD_SIZE_BITS})"
                )
        return False


class ReplaceBconv2DBitpackedDeepInPass(ReplaceBconv2DBitpackedPass):
    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_bconv2d_bin_DI)

    def match(self, op: Operator) -> bool:
        with self.using(op):
            return super().match(op) and self._input_channels % VECTOR_SIZE_BITS == 0


class ReplaceLceQuantizePass(ReplaceQuantizedOperatorPass):
    @property
    def new_opcode(self) -> OperatorCode:
        return OperatorCode(XCOREOpCodes.XC_bsign_8)

    @property
    def matching_opcode(self) -> ExternalOpCodes:
        return ExternalOpCodes.LceQuantize

    @property
    def matching_output_type(self) -> TensorType:
        return TensorType.INT32

    def match(self, op: Operator) -> bool:
        if super().match(op):
            input_shape = op.inputs[0].shape
            if len(input_shape) == 4 and input_shape[3] % WORD_SIZE_BITS == 0:
                return True
            self.logger.warning(
                f"Found LceQuantize with illegal input shape {input_shape}"
            )
        return False


class LegalizeBconv2dPass(LegalizeWeightBiasPass):
    @property
    def matching_input_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def _kernel_channel_size(self) -> int:
        # call only after custom options are set with weights shape
        return np.prod(self._op.custom_options["K"][1:])  # type: ignore

    @property
    def _overlap_size(self) -> int:
        return (
            VECTOR_SIZE_WORDS
            - 1
            - (self._kernel_channel_size // WORD_SIZE_BITS - 1) % VECTOR_SIZE_WORDS
        )

    @property
    def _fill_size(self) -> int:
        return self._overlap_size

    @staticmethod
    def __c_out_group_bounds(c_out_group: int, num_c_out: int) -> Tuple[int, int]:
        c_out_group_start = c_out_group * ACC_PERIOD
        c_out_group_end = min(num_c_out, (c_out_group + 1) * ACC_PERIOD)
        return c_out_group_start, c_out_group_end

    def mutate_weights(self, op: Operator) -> None:
        with self.using(op):
            weights = self._weights.as_array()

            num_c_out = weights.shape[0]
            num_cout_groups = ceil(num_c_out / ACC_PERIOD)

            # first we reorder the weights
            reordered_weight_channels: List[np.ndarray] = []
            for c_out_group in range(num_cout_groups):
                c_start, c_end = self.__c_out_group_bounds(c_out_group, num_c_out)
                chan_group = weights.reshape(num_c_out, -1)[c_start:c_end]
                reordered_weight_channels.extend(
                    a.ravel()
                    for a in np.split(
                        np.flip(chan_group, axis=0),
                        [
                            i * VECTOR_SIZE_WORDS
                            for i in range(
                                ceil(chan_group.shape[-1] / VECTOR_SIZE_WORDS)
                            )
                        ],
                        axis=1,
                    )
                )

            # then we need to add filler bits at the end of the last channel
            # NOTE: this means that this tensor is no longer rectangular
            reordered_weight_channels.append(
                # TODO: fix this filler value
                np.full(self._fill_size, FILLER, dtype=weights.dtype)
            )
            self._replace_weights(np.concatenate(reordered_weight_channels))

    def _calculate_overlap_correction(self, boggled_weights: np.ndarray) -> np.ndarray:
        channel_size_words = self._kernel_channel_size // WORD_SIZE_BITS
        tail_size = VECTOR_SIZE_WORDS - self._overlap_size
        overlap_correction = np.empty(self._biases.shape, dtype=np.int32)
        num_channels_out = self._biases.shape[0]
        for c_out in range(num_channels_out):
            c_out_group = c_out // ACC_PERIOD
            c_start, c_end = self.__c_out_group_bounds(c_out_group, num_channels_out)
            reversed_offset = c_out % (c_end - c_start) * tail_size
            overlap_start = c_end * channel_size_words - reversed_offset

            junk = boggled_weights[overlap_start : overlap_start + self._overlap_size]
            overlap_correction[c_out] = (
                xor_popcount(junk, np.zeros_like(junk)) - junk.size * WORD_SIZE_BITS / 2
            )
        return overlap_correction

    def mutate(self, op: Operator) -> Operator:
        with self.using(op):
            op.add_custom_options(K=get_unpacked_shape(self._weights.shape))
        # NOTE: the order of these mutations is strict
        self.mutate_weights(op)
        self.mutate_biases(op)
        op.custom_options.pop("illegal_params")
        return op


class LegalizeBconv2dInt8Pass(LegalizeBconv2dPass):
    @property
    def matching_output_type(self) -> TensorType:
        return TensorType.INT8

    @property
    def matching_opcode(self) -> XCOREOpCodes:
        return XCOREOpCodes.XC_bconv2d_int8

    @property
    def _fill_size(self) -> int:
        k_p_adjust = (
            self._kernel_channel_size // WORD_SIZE_BITS - 1
        ) % VECTOR_SIZE_WORDS + 1
        patch_loop_counter = ceil(self._kernel_channel_size / VECTOR_SIZE_BITS) - 1
        out_tail_chans = int(self._weights.shape[0] - 1) % ACC_PERIOD + 1
        fill_words = (patch_loop_counter > 0) * (
            ACC_PERIOD - out_tail_chans
        ) * VECTOR_SIZE_WORDS - k_p_adjust * out_tail_chans
        return max(fill_words, VECTOR_SIZE_WORDS)

    @staticmethod
    def __calculate_MBA(
        max_pam_exp: int, max_pab_exp: int, max_accu_exp: int
    ) -> Tuple[int, int, int]:
        accu_hat_bits = pam_hat_bits = TensorType.INT16.sizeof() * 8 - 1
        pab_hat_bits = TensorType.INT32.sizeof() * 8 - 1

        for B in reversed(range(-max_pab_exp, pab_hat_bits - max_pab_exp)):
            for A in reversed(range(-max_accu_exp, accu_hat_bits - max_accu_exp)):
                M = B - A
                if -max_pam_exp <= M < pam_hat_bits - max_pam_exp:
                    return M, B, A
        raise ValueError("quantized exponents cannot be determined")

    def _calculate_quantization_parameters(
        self, adjusted_pam: np.ndarray, adjusted_pab: np.ndarray
    ) -> Tuple[int, int, Tuple[int, int, int]]:
        max_pam_exp = np.max(np.frexp(adjusted_pam)[1])
        max_pab_exp = np.max(np.frexp(adjusted_pab)[1])
        max_accu_exp = np.frexp(self._kernel_channel_size / 2)[1]
        M, B, A = self.__calculate_MBA(max_pam_exp, max_pab_exp, max_accu_exp)

        adjusted_B = min(15 - max_pab_exp, B)
        assert 15 > B - adjusted_B
        bias_multiplier = 2 ** (B - adjusted_B)  # this is not so simple

        accu_shr = -A
        final_shr = B - 8

        return M, adjusted_B, (bias_multiplier, accu_shr, final_shr)

    def mutate_biases(self, op: Operator) -> None:
        with self.using(op):
            # first we adjust pam/pab as the larq kernel's output transform requires
            output_scale = self._output.quantization["scale"][0]
            output_zero_point = self._output.quantization["zero_point"][0]
            post_act_mult_float = self._op.inputs[2].as_array()
            post_act_bias_float = self._op.inputs[3].as_array()

            output_trf_pam = -post_act_mult_float / output_scale
            output_trf_pab = (
                post_act_bias_float / output_scale
                - output_trf_pam * self._kernel_channel_size
                + output_zero_point
            )

            # then adjust pam/pad as required by our kernels
            weights = self._weights.as_array()  # already boggled
            adjusted_pam = -2 * output_trf_pam
            adjusted_pab = output_trf_pab + output_trf_pam * (
                self._kernel_channel_size
                - 2 * self._calculate_overlap_correction(weights)
            )

            # calculate quantization parameters as required by the kernel
            (M, adjusted_B, q_params) = self._calculate_quantization_parameters(
                adjusted_pam, adjusted_pab
            )
            op.add_custom_options(q_params=q_params)

            # then quantize the post activation parameters
            pam_q = np.round(adjusted_pam * 2.0 ** M)
            post_act_mult_quant = pam_q.astype(np.int16)
            assert np.all(post_act_mult_quant == pam_q)

            # TODO: fix this so there is no need to clip
            pab_q = np.round(adjusted_pab * 2.0 ** adjusted_B)
            post_act_bias_quant = np.clip(
                pab_q, np.iinfo(np.int16).min, np.iinfo(np.int16).max
            ).astype(np.int16)
            if np.any(post_act_bias_quant != pab_q):
                self.logger.warning("clipped post_act_bias_quant")

            # create and populate new threshpost_act_multolds tensor
            new_pam_tensor = self._op.subgraph.create_tensor(
                f"{self._op.name}/post_act_mult",
                TensorType.INT16,
                self._op.inputs[2].shape,
                consumers=[self._op],
            )
            new_pam_tensor.buffer.data = post_act_mult_quant

            # replace old pam tensor
            self._op.inputs[2].consumers.remove(self._op)
            self._op.inputs[2] = new_pam_tensor

            # create and populate new threshpost_act_multolds tensor
            new_pab_tensor = self._op.subgraph.create_tensor(
                f"{self._op.name}/post_act_bias",
                TensorType.INT16,
                self._op.inputs[3].shape,
                consumers=[self._op],
            )
            new_pab_tensor.buffer.data = post_act_bias_quant

            # replace old pam tensor
            self._op.inputs[3].consumers.remove(self._op)
            self._op.inputs[3] = new_pab_tensor


class LegalizeBconv2dInt8DeepInDeepOutPass(LegalizeBconv2dInt8Pass):
    @property
    def _overlap_size(self) -> int:
        return 0

    @property
    def matching_opcode(self) -> XCOREOpCodes:
        return XCOREOpCodes.XC_bconv2d_int8_DIDO


class LegalizeBconv2dBitpackedPass(LegalizeBconv2dPass):
    @property
    def matching_output_type(self) -> TensorType:
        return TensorType.INT32

    @property
    def matching_opcode(self) -> XCOREOpCodes:
        return XCOREOpCodes.XC_bconv2d_bin

    def mutate_biases(self, op: Operator) -> None:
        with self.using(op):
            thresholds = self._biases.as_array()
            weights = self._weights.as_array()  # already boggled

            # first we need to calculate a correction term
            # due to how our HW popcount differs from the Larq reference
            popcount_correction = self._kernel_channel_size / 2

            # second we need to calculate correction terms
            # due to how we handle incomplete weights regsiters
            # (the data register is padded with zeros, so the loaded kernel
            # coeffs can have some junk loaded, and we correct that)
            overlap_correction = self._calculate_overlap_correction(weights)
            thresholds += np.int32(overlap_correction - popcount_correction)

            # boggle the lower and higher 2 bytes in every ACC_PERIOD consecutive value
            thresholds = np.concatenate(
                [
                    np.frombuffer(
                        np.frombuffer(cgroup.tostring(), dtype=np.int16)
                        .reshape(ACC_PERIOD, 2)
                        .T.tostring(),
                        dtype=np.int32,
                    )
                    for cgroup in thresholds.reshape(
                        thresholds.shape[0] // ACC_PERIOD, ACC_PERIOD
                    )
                ]
            )

            # create and populate new thresholds tensor
            new_thresholds = self._op.subgraph.create_tensor(
                f"{self._op.name}/thresholds",
                TensorType.INT32,
                thresholds.shape,
                consumers=[self._op],
            )
            new_thresholds.buffer.data = thresholds

            # replace old tensor
            self._op.inputs[2].consumers.remove(self._op)
            self._op.inputs[2] = new_thresholds


class LegalizeBconv2dBitpackedDeepInPass(LegalizeBconv2dBitpackedPass):
    @property
    def matching_opcode(self) -> XCOREOpCodes:
        return XCOREOpCodes.XC_bconv2d_bin_DI

    @property
    def _overlap_size(self) -> int:
        return 0


# Split out padding to a separate op from BConv
# TODO: this currently only matches with XC_bconv2d_*
# but going forward might like to extend this to other conv ops
# and make it a more general pass for all convolutions.
class LegalizeXCBconv2DPaddingPass(OperatorMatchingPass):
    @property
    def _strides(self) -> Tuple[int, int]:
        return self._op.custom_options["stride"]

    @property
    def _padding(self) -> Padding:
        return self._op.custom_options["padding"]

    MATCHING_OPCODES = XC_BCONV2D_OPCODES

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and op.operator_code.code in self.MATCHING_OPCODES
            and "padding" in op.custom_options
        )

    def mutate(self, op: Operator) -> Operator:
        padding = Padding(op.custom_options.pop("padding"))
        if padding is Padding.VALID:
            return op

        old_input = op.inputs[0]

        # calculate paddings
        with self.using(op):
            input_and_strides = old_input.shape[1:3], self._strides

        paddings = np.int32(
            [
                (0, 0),
                *calculate_same_padding(*input_and_strides, op.inputs[1].shape[1:3]),
                (0, 0),
            ]
        )

        # return early if mode is SAME, but has no effect
        if np.all(paddings == 0):
            return op

        subgraph = op.subgraph

        # Construct paddings parameter tensor and padded input tensor
        padding_tensor = subgraph.create_tensor(
            f"{op.name}/paddings", TensorType.INT32, shape=paddings.shape
        )
        padding_tensor.buffer.data = paddings

        padded_shape = tuple(
            int(size + sum(pads)) for size, pads in zip(old_input.shape, paddings)
        )
        padded_input_tensor = subgraph.create_tensor(
            f"{op.name}/input", TensorType.INT32, shape=padded_shape, consumers=[op],
        )

        # create new PAD op and inject it before the convolution
        pad_op = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.PAD),
            inputs=[old_input, padding_tensor],
            outputs=[padded_input_tensor],
        )
        subgraph.insert_operator(op, pad_op)

        # Cut connection from old input to the op
        old_input.consumers.remove(op)
        op.inputs[0] = padded_input_tensor

        return op
