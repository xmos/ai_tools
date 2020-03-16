#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_INPUTS = 36
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = 9
DEFAULT_POOL_HEIGHT = 3
DEFAULT_POOL_WIDTH = 5
DEFAULT_POOL_SIZE = (DEFAULT_POOL_HEIGHT, DEFAULT_POOL_WIDTH)
DEFAULT_PADDING = "valid"
DEFAULT_STRIDE_HEIGHT = 1
DEFAULT_STRIDE_WIDTH = 2
DEFAULT_STRIDES = (DEFAULT_STRIDE_HEIGHT, DEFAULT_STRIDE_WIDTH)
DEFAULT_PATH = Path(__file__).parent.joinpath("debug", "avgpool2d").resolve()


class AvgPool2D(common.OpTestDefaultModel):
    def build_core_model(self, height, width, input_channels, *,
                         pool_size, strides, padding, input_init):
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
                tf.keras.layers.AveragePooling2D(
                    pool_size=pool_size, strides=strides, padding=padding,
                    input_shape=(height, width, input_channels))
            ]
        )


def main(raw_args=None):
    parser = common.OpTestPoolParser(defaults={
        "path": DEFAULT_PATH,
        "inputs": DEFAULT_INPUTS,
        "height": DEFAULT_HEIGHT,
        "width": DEFAULT_WIDTH,
        "padding": DEFAULT_PADDING,
        "strides": DEFAULT_STRIDES,
        "pool_size": DEFAULT_POOL_SIZE,
        'inits': {
            'input_init': {'type': common.OpTestInitializers.UNIF}
        }
    })
    args = parser.parse_args(raw_args)

    model = AvgPool2D("avgpool2d", args.path)
    model.build(args.height, args.width, args.inputs,
                padding=args.padding,
                pool_size=args.pool_size,
                strides=args.strides,
                **args.inits)
    model.run()


if __name__ == "__main__":
    main()
