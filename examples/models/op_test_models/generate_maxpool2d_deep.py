#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_INPUTS = 32
DEFAULT_HEIGHT = 4
DEFAULT_WIDTH = DEFAULT_HEIGHT

DEFAULT_POOL_SIZE = 2
DEFAULT_PADDING = "valid"
DEFAULT_STRIDES = 2
DEFAULT_PATH = Path(__file__).parent.joinpath("debug",
                                              "maxpool_2d_deep").resolve()


class MaxPool2d(common.DefaultOpTestModel):
    def build(self, height, width, input_channels, *, pool_size, strides,
              padding):
        assert input_channels % 32 == 0, "# of input channels must be multiple of 32"
        assert height % 2 == 0, "height must be even"
        assert width % 2 == 0, "width must be even"
        assert pool_size == 2, "pool size must be 2"
        assert strides == 2, "stride must be 2"
        assert padding.lower() == "valid", "padding mode must be valid"
        super().build()

        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.MaxPool2D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    input_shape=(height, width, input_channels),
                )
            ],
        )
        # Compilation
        self.core_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        # Show summary
        self.core_model.summary()


def main(path=DEFAULT_PATH,
         *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         pool_size=DEFAULT_POOL_SIZE,
         padding=DEFAULT_PADDING,
         strides=DEFAULT_STRIDES):
    # Instantiate model
    test_model = MaxPool2d("maxpool2d_deep", Path(path))
    # Build model and compile
    test_model.build(height,
                     width,
                     input_channels,
                     pool_size=pool_size,
                     strides=strides,
                     padding=padding)
    test_model.run()


if __name__ == "__main__":
    parser = common.OpTestPoolStridesParser(
        defaults={
            "path": DEFAULT_PATH,
            "inputs": DEFAULT_INPUTS,
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT,
            "padding": DEFAULT_PADDING,
            "strides": DEFAULT_STRIDES,
            "pool_size": DEFAULT_POOL_SIZE
        })
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    strides_pool = common.strides_pool_arg_handler(args)

    main(
        path=args.path,
        input_channels=args.inputs,
        height=args.height,
        width=args.width,
        padding=args.padding,
        **strides_pool,
    )
