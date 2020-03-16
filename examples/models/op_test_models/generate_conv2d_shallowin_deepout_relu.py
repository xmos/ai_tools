#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_INPUTS = 3
DEFAULT_OUTPUTS = 16
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = 'same'
DEFAULT_PATH = Path(__file__).parent.joinpath(
    'debug', 'conv2d_shallowin_deepout_relu').resolve()


class Conv2DShallowinDeepoutRelu(common.OpTestDeepoutConvModel):
    def build_core_model(self, *args, **kwargs):
        K_w, input_channels = args[1], args[4]
        assert input_channels <= 4, "Number of input channels must be at most 4"
        assert K_w <= 8, "Kernel width must be at most 8"
        super().build_core_model(*args, **kwargs)


def main(raw_args=None):
    parser = common.OpTestConvParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'outputs': DEFAULT_OUTPUTS,
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT,
        'padding': DEFAULT_PADDING,
        'kernel_width': DEFAULT_KERNEL_WIDTH,
        'kernel_height': DEFAULT_KERNEL_HEIGHT,
        'inits': {
            'input_init': {'type': common.OpTestInitializers.UNIF},
            'weight_init': {'type': common.OpTestInitializers.UNIF},
            'bias_init': {'type': common.OpTestInitializers.CONST}
        }
    })
    args = parser.parse_args(raw_args)

    model = Conv2DShallowinDeepoutRelu('conv2d_shallowin_deepout_relu', args.path)
    model.build(args.kernel_height, args.kernel_width,
                args.height, args.width,
                args.inputs, args.outputs,
                padding=args.padding, **args.inits)


if __name__ == "__main__":
    main()
