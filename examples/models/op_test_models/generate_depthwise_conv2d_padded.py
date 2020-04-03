#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common


DEFAULT_OUTPUTS = 8
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = (1, 1, 1, 1)
DEFAULT_PATH = Path(__file__).parent.joinpath(
    'debug', 'depthwise_conv2d_padded').resolve()


class DepthwiseConv2DPadded(common.OpTestDefaultModel):
    def build_core_model(
            self, K_h, K_w, height, width, output_channels, *,
            padding, strides, **inits):
        assert output_channels % 4 == 0, "# of output channels must be multiple of 4"
        self.input_init = inits['input_init']
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.ZeroPadding2D(padding=padding,
                                              input_shape=(height, width, output_channels)),
                tf.keras.layers.DepthwiseConv2D(kernel_size=(K_h, K_w),
                                                depth_multiplier=1,
                                                padding='valid',
                                                strides=strides,
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
        'padding': None,
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
    parser.add_argument(
        "-pd", "--padding", type=int, nargs=4, default=DEFAULT_PADDING,
        help="Zero padding for each image edge with order (top, bottom, left, right)."
    )
    parser.add_argument(  # TODO: use the a better parser for this after the conv2d enhancements
        "-st", "--strides", nargs=2, type=int, default=[1, 1],
        help=f"Strides, vertical first",
    )
    args = parser.parse_args(raw_args)
    args.strides = tuple(args.strides)  # TODO: fix this
    args.padding = (args.padding[:2], args.padding[2:])

    model = DepthwiseConv2DPadded('depthwise_conv2d_padded', args.path)
    model.build(args.kernel_height, args.kernel_width,
                args.height, args.width,
                args.outputs,
                padding=args.padding,
                strides=args.strides,
                **args.inits)
    model.run()


if __name__ == "__main__":
    main()
