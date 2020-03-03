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


class Conv2dDeepinDeepoutRelu(common.OpTestDeepoutConvModel):
    def build_core_model(self, *args, **kwargs):
        input_channels = args[4]
        K_h, K_w = args[0], args[1]
        assert input_channels % 32 == 0, "# of input channels must be multiple of 32"
        assert K_h != 1 or K_w != 1, "1x1 kernel is not allowed for DIDO testing"
        super().build_core_model(*args, **kwargs)


def main(path=DEFAULT_PATH, *,
         num_threads=DEFAULT_NUM_THREADS,
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
        'name': 'conv2d_deepin_deepout_relu',
        'path': path if path else DEFAULT_PATH
    }
    common.run_main_conv(model=Conv2dDeepinDeepoutRelu(**kwargs),
                         num_threads=num_threads,
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
    parser = common.OpTestConvParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'outputs': DEFAULT_OUTPUTS,
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT,
        'padding': DEFAULT_PADDING,
        'kernel_width': DEFAULT_KERNEL_WIDTH,
        'kernel_height': DEFAULT_KERNEL_HEIGHT
    })
    parser.add_argument(
        '-par', '--par_num_threads', type=int, default=DEFAULT_NUM_THREADS,
        help='Number of parallel threads for xcore.ai optimization.')
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    initializers = common.initializer_args_handler(args)

    main(path=args.path,
         num_threads=args.par_num_threads,
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
