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


class Conv2D1x1(common.DefaultOpTestConvModel):
    def build_core_model(self, *args, **kwargs):
        K_h, K_w, _, _, input_channels, output_channels = args
        assert K_h == 1, "Kernel height must be 1"
        assert K_w == 1, "Kernel width must be 1"
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        assert output_channels % 4 == 0, "# of output channels must be multiple of 4"
        super().build_core_model(*args, **kwargs)


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         output_channels=DEFAULT_OUTPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         padding=DEFAULT_PADDING,
         bias_init=common.DEFAULT_CONST_INIT,
         weight_init=common.DEFAULT_UNIF_INIT,
         input_init=common.DEFAULT_UNIF_INIT):
    kwargs = {
        'name': 'conv2d_deepin_deepout_relu',
        'path': path if path else DEFAULT_PATH
    }
    common.run_main_conv(model=Conv2D1x1(**kwargs),
                         num_threads=None,
                         input_channels=input_channels,
                         output_channels=output_channels,
                         height=height,
                         width=width,
                         K_h=1,
                         K_w=1,
                         padding=padding,
                         bias_init=bias_init,
                         weight_init=weight_init,
                         input_init=input_init)


class OpTestConv1x1Parser(common.OpTestParserInitializerMixin, common.OpTestDimParser):
    def __init__(self, *args, defaults, **kwargs):
        super().__init__(*args, defaults=defaults, **kwargs)
        self.add_argument(
            "-out", "--outputs", type=int, default=defaults["outputs"],
            help="Number of output channels",
        )
        self.add_argument(
            "-st", "--strides", nargs="+", type=int, default=argparse.SUPPRESS,
            help=f"Strides, vertical first (default: {defaults['strides']})",
        )
        self.add_initializers()


if __name__ == "__main__":
    parser = OpTestConv1x1Parser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'outputs': DEFAULT_OUTPUTS,
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT,
        'padding': DEFAULT_PADDING,
        'strides': DEFAULT_STRIDES,
    })
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    initializers = common.initializer_args_handler(args)
    strides_pool = common.strides_pool_arg_handler(args)

    main(path=args.path,
         input_channels=args.inputs,
         output_channels=args.outputs,
         height=args.height,
         width=args.width,
         padding=args.padding,
         bias_init=initializers['bias_init'],
         weight_init=initializers['weight_init'],
         input_init=initializers['input_init'])
