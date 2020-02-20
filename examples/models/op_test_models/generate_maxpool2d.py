#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
import tensorflow as tf

from generate_avgpool2d import (
    DefaultPool2DModel, DefaultPool2DParser, strides_pool_arg_handler,
    DEFAULT_INPUTS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_POOL_HEIGHT,
    DEFAULT_POOL_WIDTH,
    DEFAULT_POOL_SIZE,
    DEFAULT_PADDING,
    DEFAULT_STRIDE_HEIGHT,
    DEFAULT_STRIDE_WIDTH,
    DEFAULT_STRIDES,
)

DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'maxpool2d').resolve()


class MaxPool2d(DefaultPool2DModel):
    def build_core_model(self, height, width, input_channels,
                         *, pool_size, strides, padding):
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        if padding.lower() == 'same':
            assert (height % 2 == 0 and width % 2 == 0
                    and pool_size[0] == 2 and pool_size[1] == 2
                    and strides[0] == 2 and strides[1] == 2), "same padding is only allowed for the common 2x2 case"
        else:
            assert padding.lower() == 'valid', f"invalid padding mode '{padding}'"

        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.MaxPool2D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    input_shape=(height, width, input_channels))
            ]
        )


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
         pool_size=DEFAULT_POOL_SIZE,
         strides=DEFAULT_STRIDES,
         padding=DEFAULT_PADDING):
    model = MaxPool2d('maxpool2d', Path(path))
    model.build(height, width, input_channels,
                pool_size=pool_size,
                strides=strides,
                padding=padding)
    model.run()


if __name__ == "__main__":
    parser = DefaultPool2DParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'height': DEFAULT_HEIGHT,
        'width': DEFAULT_WIDTH
    })
    parser.add_argument(
        '-st', '--strides', nargs='+', type=int, default=argparse.SUPPRESS,
        help="Strides, vertical first "
             f"(default: {DEFAULT_STRIDE_HEIGHT} {DEFAULT_STRIDE_WIDTH})")
    parser.add_argument(
        '-po', '--pool_size', nargs='+', type=int, default=argparse.SUPPRESS,
        help="Pool size:, vertical first "
             f"(default: {DEFAULT_POOL_HEIGHT} {DEFAULT_POOL_WIDTH})")
    parser.add_argument(
        '-pd', '--padding', type=str, default=DEFAULT_PADDING,
        help='Padding mode')
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    strides_pool = strides_pool_arg_handler(args)

    main(path=args.path,
         input_channels=args.inputs,
         height=args.height, width=args.width,
         padding=args.padding,
         **strides_pool
         )
