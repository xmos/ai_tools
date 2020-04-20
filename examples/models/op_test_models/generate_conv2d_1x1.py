#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_INPUTS = 4
DEFAULT_OUTPUTS = 8
DEFAULT_HEIGHT = 4
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_PADDING = 'same'
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'conv2d_1x1').resolve()
DEFAULT_NUM_THREADS = 1


class Conv2D1x1(common.OpTestDefaultConvModel):
    def build_core_model(self, *args, **kwargs):
        K_h, K_w, _, _, input_channels, output_channels = args
        assert K_h == 1, "Kernel height must be 1"
        assert K_w == 1, "Kernel width must be 1"
        # TODO: move these to a parent class after the conv2d enhancements
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        assert output_channels % 4 == 0, "# of output channels must be multiple of 4"
        super().build_core_model(*args, **kwargs)


def main(raw_args=None):
    parser = common.OpTestConvParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'outputs': DEFAULT_OUTPUTS,
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT,
        'padding': DEFAULT_PADDING,
        'kernel_width': 1,
        'kernel_height': 1,
        'inits': {
            'input_init': {'type': common.OpTestInitializers.UNIF},
            'weight_init': {'type': common.OpTestInitializers.UNIF},
            'bias_init': {'type': common.OpTestInitializers.CONST}
        }
    })
    parser.add_argument(
        "-kh", "--kernel_height", type=int, default=1, choices=[1],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-kw", "--kernel_width", type=int, default=1, choices=[1],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '-par', '--num_threads', type=int, default=DEFAULT_NUM_THREADS,
        help='Number of parallel threads for xcore.ai optimization.')
    args = parser.parse_args(raw_args)

    model = Conv2D1x1('conv2d_1x1', args.path)
    model.build(1, 1,
                args.height, args.width,
                args.inputs, args.outputs,
                padding=args.padding, **args.inits)
    model.run(num_threads=args.num_threads)


if __name__ == "__main__":
    main()
