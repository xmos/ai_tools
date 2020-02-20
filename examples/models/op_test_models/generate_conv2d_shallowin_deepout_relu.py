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


class Conv2dShallowinDeepoutRelu(common.DefaultOpTestConvModel):
    def build_core_model(self, K_h, K_w, height, width, input_channels, output_channels,
              *, padding, bias_init, weight_init, input_init):
        assert input_channels <= 4, "Number of input channels must be at most 4"
        assert K_w <= 8, "Kernel width must be at most 8"
        super().build_core_model(K_h,
                      K_w,
                      height,
                      width,
                      input_channels,
                      output_channels,
                      padding=padding,
                      bias_init=bias_init,
                      weight_init=weight_init,
                      input_init=input_init)


def main(path=DEFAULT_PATH,
         *,
         input_channels=DEFAULT_INPUTS,
         output_channels=DEFAULT_OUTPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         K_h=DEFAULT_KERNEL_HEIGHT,
         K_w=DEFAULT_KERNEL_WIDTH,
         padding=DEFAULT_PADDING,
         bias_init=common.DEFAULT_CONST_INIT,
         weight_init=common.DEFAULT_UNIF_INIT,
         input_init=common.DEFAULT_UNIF_INIT):
    kwargs = {
        'name': 'conv2d_shallowin_deepout_relu',
        'path': path if path else DEFAULT_PATH
    }
    common.run_main_conv(model=Conv2dShallowinDeepoutRelu(**kwargs),
                         num_threads=None,
                         input_channels=input_channels,
                         output_channels=output_channels,
                         height=height,
                         width=width,
                         K_h=K_h,
                         K_w=K_w,
                         padding=padding,
                         bias_init=bias_init,
                         weight_init=weight_init,
                         input_init=input_init)


if __name__ == "__main__":
    parser = common.OpTestConvParser(
        defaults={
            'path': DEFAULT_PATH,
            'inputs': DEFAULT_INPUTS,
            'outputs': DEFAULT_OUTPUTS,
            'width': DEFAULT_WIDTH,
            'height': DEFAULT_HEIGHT,
            'padding': DEFAULT_PADDING,
            'kernel_width': DEFAULT_KERNEL_WIDTH,
            'kernel_height': DEFAULT_KERNEL_HEIGHT
        })
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    initializers = common.initializer_args_handler(args)

    main(path=args.path,
         input_channels=args.inputs,
         output_channels=args.outputs,
         K_h=args.kernel_height,
         K_w=args.kernel_width,
         height=args.height,
         width=args.width,
         padding=args.padding,
         bias_init=initializers['bias_init'],
         weight_init=initializers['weight_init'],
         input_init=initializers['input_init'])
