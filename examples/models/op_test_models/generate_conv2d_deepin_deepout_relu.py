#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_INPUTS = 32
DEFAULT_OUTPUTS = 16
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = 'same'
DEFAULT_PATH = Path(__file__).parent.joinpath(
    'debug', 'conv2d_deepin_deepout_relu').resolve()
DEFAULT_NUM_THREADS = 1


class Conv2DDeepinDeepoutRelu(common.OpTestDeepoutConvModel):
    def build_core_model(self, *args, **kwargs):
        input_channels = args[4]
        K_h, K_w = args[0], args[1]
        assert input_channels % 32 == 0, "# of input channels must be multiple of 32"
        assert K_h != 1 or K_w != 1, "1x1 kernel is not allowed for DIDO testing"
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
            'input_init': {
                'type': common.OpTestInitializers.UNIF,
                'help': "Initializer for input data distribution."
            },
            'weight_init': {
                'type': common.OpTestInitializers.UNIF,
                'help': "Initializer for weight distribution."
            },
            'bias_init': {
                'type': common.OpTestInitializers.CONST,
                'help': "Initializer for bias distribution."
            }
        }
    })
    parser.add_argument(
        '-par', '--num_threads', type=int, default=DEFAULT_NUM_THREADS,
        help='Number of parallel threads for xcore.ai optimization.')
    args = parser.parse_args(raw_args)

    model = Conv2DDeepinDeepoutRelu('conv2d_deepin_deepout_relu', args.path)
    model.run(num_threads=args.num_threads,
              input_channels=args.inputs,
              output_channels=args.outputs,
              height=args.height,
              width=args.width,
              K_h=args.kernel_height,
              K_w=args.kernel_width,
              padding=args.padding,
              **args.inits)


if __name__ == "__main__":
    main()
