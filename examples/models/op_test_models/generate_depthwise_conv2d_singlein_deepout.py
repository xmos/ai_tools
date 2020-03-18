#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_OUTPUTS = 16
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = 'same'
DEFAULT_PATH = Path(__file__).parent.joinpath(
    'debug', 'depthwise_conv2d_singlein_deepout').resolve()


class DepthwiseConv2DSingleinDeepout(common.OpTestDefaultModel):
    def build_core_model(
            self, K_h, K_w, height, width, output_channels, *,
            padding, **inits):
        assert output_channels % 16 == 0, "# of output channels must be multiple of 16"
        assert K_w <= 8, "Kernel width must be at most 8"
        # TODO: remove these constraints when conv2d improvements are ready
        assert K_h % 2 == 1, "kernel height must be odd"
        assert K_w % 2 == 1, "kernel width must be odd"
        self.input_init = inits['input_init']
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.DepthwiseConv2D(kernel_size=(K_h, K_w),
                                                depth_multiplier=output_channels,
                                                padding=padding,
                                                input_shape=(height, width, 1),
                                                bias_initializer=inits['bias_init'],
                                                depthwise_initializer=inits['weight_init'])
            ]
        )


def main(raw_args=None):
    parser = common.OpTestConvParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': -1,
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
    parser.add_argument(
        "-in", "--inputs", type=int, default=-1, choices=[-1],
        help=argparse.SUPPRESS
    )
    args = parser.parse_args(raw_args)

    model = DepthwiseConv2DSingleinDeepout('depthwise_conv2d_singlein_deepout', args.path)
    model.build(args.kernel_height, args.kernel_width,
                args.height, args.width,
                args.outputs,
                padding=args.padding,
                **args.inits)
    model.run()


if __name__ == "__main__":
    main()
