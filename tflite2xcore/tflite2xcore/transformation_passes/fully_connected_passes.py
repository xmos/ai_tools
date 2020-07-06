# Copyright (c) 2020, XMOS Ltd, All rights reserved

import numpy as np

from tflite2xcore.xcore_schema import (
    TensorType,
    BuiltinOpCodes,
    OperatorCode,
    XCOREOpCodes,
)
from tflite2xcore.utils import WORD_SIZE
from .transformation_passes import (
    ReplaceWeightBiasOperatorPass,
    QuantizedOperatorMatchingPass,
    LegalizeXCWeightBiasPass,
    OperatorMatchingPass,
)
from tflite2xcore.parallelization import ChannelGroupSlicePlanner, CHANNEL_GROUP_SIZE
from tflite2xcore.xlogging import log_method_output


class ReplaceFullyConnectedPass(ReplaceWeightBiasOperatorPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.FULLY_CONNECTED

    @property
    def new_opcode(self):
        return OperatorCode(XCOREOpCodes.XC_fc_deepin_anyout)


class LegalizeXCFullyConnectedPass(LegalizeXCWeightBiasPass):
    @property
    def matching_opcode(self):
        return XCOREOpCodes.XC_fc_deepin_anyout

    @property
    def _OUTPUT_BITS(self):
        return 16

    @log_method_output()
    def _zero_point_bias(self):
        return np.sum(self._weights.numpy * self._input_zero_point, axis=1)

    def mutate_weights(self, op):
        with self.using(op):
            # zero_padding weight tensor
            col_pad = WORD_SIZE - 1 - (self._weights.shape[1] - 1) % WORD_SIZE
            arr = np.pad(
                self._weights.numpy.astype(np.int8), pad_width=[(0, 0), (0, col_pad)]
            )

            self._replace_weights(arr)
            self._log_weights()

    def mutate_output(self, op):
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
        super().mutate(op)
        self.add_requantize(op)
        self.mutate_output(op)


class ParallelizeFullyConnectedPass(OperatorMatchingPass):
    def __init__(self, *args, num_threads=None, forced=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_threads = num_threads or 1
        assert isinstance(self.max_threads, int)
        assert self.max_threads > 0
        self.forced = forced

    def match(self, op):
        if (
            super().match(op)
            and op.operator_code.code == XCOREOpCodes.XC_fc_deepin_anyout
        ):
            return "par" not in op.custom_options

    def mutate(self, op):
        _, Cout = op.outputs[0].shape
        planner = ChannelGroupSlicePlanner(
            int(Cout), num_threads=self.max_threads, forced=self.forced
        )
        plan = planner.find_optimal_plan()
        plan.num_threads = min(plan.num_threads, len(plan.changrp_slices))

        op.add_custom_options(par=plan.to_dict())


class ParallelizeRequant16To8Pass(OperatorMatchingPass):
    def __init__(self, *args, num_threads=None, forced=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_threads = num_threads or 1
        assert isinstance(self.max_threads, int)
        assert self.max_threads > 0
        self.forced = forced

    def match(self, op):
        if (
            super().match(op)
            and op.operator_code.code == XCOREOpCodes.XC_requantize_16_to_8
        ):
            return "par" not in op.custom_options

    def mutate(self, op):
        _, Cout = op.outputs[0].shape
        planner = ChannelGroupSlicePlanner(
            int(Cout), num_threads=self.max_threads, forced=self.forced
        )
        plan = planner.find_optimal_plan()
        plan.num_threads = min(plan.num_threads, len(plan.changrp_slices))

        op.add_custom_options(par=plan.to_dict())


class ScratchMemoryFullyConnectedPass(OperatorMatchingPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def match(self, op):
        if (
            super().match(op)
            and op.operator_code.code == XCOREOpCodes.XC_fc_deepin_anyout
        ):
            return "mem" not in op.custom_options

    def mutate(self, op):
        _, Bv, Bl = op.inputs[2].shape

        if "par" in op.custom_options:
            max_cg_size = max(
                [cg[1] - cg[0] + 1 for cg in op.custom_options["par"]["cg"]]
            )
        else:
            max_cg_size = CHANNEL_GROUP_SIZE

        weights_scratch_size = max_cg_size
        bias_scratch_size = Bv * Bl * op.inputs[2].type.to_bytes()

        op.add_custom_options(mem=[weights_scratch_size, bias_scratch_size])

