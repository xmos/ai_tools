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
DEFAULT_STRIDE_HEIGHT = 1
DEFAULT_STRIDE_WIDTH = DEFAULT_STRIDE_HEIGHT
DEFAULT_STRIDES = (DEFAULT_STRIDE_HEIGHT, DEFAULT_STRIDE_WIDTH)
DEFAULT_PADDING = 'same'
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'conv2d_1x1').resolve()


class Conv2D1x1(common.OpTestDefaultConvModel):
    def build_core_model(self, *args, **kwargs):
        K_h, K_w, _, _, input_channels, output_channels = args
        assert K_h == 1, "Kernel height must be 1"
        assert K_w == 1, "Kernel width must be 1"
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        assert output_channels % 4 == 0, "# of output channels must be multiple of 4"
        super().build_core_model(*args, **kwargs)


def main(raw_args=None):
    parser = common.OpTestImgParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT,
        'padding': DEFAULT_PADDING,
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
        "-out", "--outputs", type=int, default=DEFAULT_OUTPUTS,
        help="Number of output channels",
    )
    args = parser.parse_args(raw_args)
    utils.set_gpu_usage(False, args.verbose)

    model = Conv2D1x1('conv2d_1x1', args.path)
    model.run(num_threads=None,
              input_channels=args.inputs,
              output_channels=args.outputs,
              height=args.height,
              width=args.width,
              K_h=1,
              K_w=1,
              padding=args.padding,
              **args.inits)


if __name__ == "__main__":
    main()
