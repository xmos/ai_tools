#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import numpy as np
import op_test_models_common as common

DEFAULT_INPUTS = 3
DEFAULT_OUTPUTS = 16
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = "same"
DEFAULT_PATH = Path(__file__).parent.joinpath("debug", "conv2d_shallowin").resolve()


class Conv2DShallowin(common.OpTestDefaultConvModel):
    def build_core_model(self, *args, **kwargs):
        K_h, K_w, _, _, input_channels, output_channels = args
        assert output_channels % 4 == 0, "# of output channels must be multiple of 4"
        assert (
            K_h != 1 or K_w != 1
        ), "1x1 kernel is not allowed for shallowin conv2d testing"

        padded_inputs = np.ceil(input_channels / 4) * 4
        assert (
            padded_inputs * K_w <= 32
        ), f"product of padded inputs count and kernel width must be at most 32"

        super().build_core_model(*args, **kwargs)


def main(raw_args=None):
    parser = common.OpTestConvParser(
        defaults={
            "path": DEFAULT_PATH,
            "inputs": DEFAULT_INPUTS,
            "outputs": DEFAULT_OUTPUTS,
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT,
            "padding": DEFAULT_PADDING,
            "kernel_width": DEFAULT_KERNEL_WIDTH,
            "kernel_height": DEFAULT_KERNEL_HEIGHT,
            "inits": {
                "input_init": {"type": common.OpTestInitializers.UNIF},
                "weight_init": {"type": common.OpTestInitializers.UNIF},
                "bias_init": {"type": common.OpTestInitializers.CONST},
            },
        }
    )
    args = parser.parse_args(raw_args)

    model = Conv2DShallowin("conv2d_shallowin", args.path)
    model.build(
        args.kernel_height,
        args.kernel_width,
        args.height,
        args.width,
        args.inputs,
        args.outputs,
        padding=args.padding,
        **args.inits,
    )
    model.run()


if __name__ == "__main__":
    main()
