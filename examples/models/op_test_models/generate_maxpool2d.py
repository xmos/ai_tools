#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

from generate_avgpool2d import (
    DEFAULT_INPUTS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_POOL_SIZE,
    DEFAULT_PADDING,
    DEFAULT_STRIDES,
)

DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'maxpool2d').resolve()


class MaxPool2d(common.DefaultOpTestModel):
    def build_core_model(self, height, width, input_channels, *, pool_size,
                         strides, padding, input_init):
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        if padding.lower() == 'same':
            assert (height % 2 == width % 2 == 0
                    and pool_size[0] == pool_size[1] == 2
                    and strides[0] == strides[1] == 2
                    ), "same padding is only allowed for the common 2x2 case"
        else:
            assert padding.lower() == 'valid', f"invalid padding mode '{padding}'"
        self.input_init = input_init
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.MaxPool2D(
                    pool_size=pool_size, strides=strides, padding=padding,
                    input_shape=(height, width, input_channels))
            ]
        )


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         pool_size=DEFAULT_POOL_SIZE,
         strides=DEFAULT_STRIDES,
         padding=DEFAULT_PADDING,
         input_init=common.DEFAULT_UNIF_INIT):
    model = MaxPool2d('maxpool2d', Path(path))
    model.build(height, width, input_channels,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                input_init=input_init)
    model.run()


if __name__ == "__main__":
    parser = common.OpTestPoolParser(defaults={
        "path": DEFAULT_PATH,
        "inputs": DEFAULT_INPUTS,
        "height": DEFAULT_HEIGHT,
        "width": DEFAULT_WIDTH,
        "padding": DEFAULT_PADDING,
        "strides": DEFAULT_STRIDES,
        "pool_size": DEFAULT_POOL_SIZE,
        'inits': {'input_init': common.OpTestInitializers.UNIF}
    })
    args = parser.parse_args()
    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    main(path=args.path,
         input_channels=args.inputs,
         height=args.height,
         width=args.width,
         padding=args.padding,
         **args.strides_pool,
         input_init=args.inits['input_init'])
