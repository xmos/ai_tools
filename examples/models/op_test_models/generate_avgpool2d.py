#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from abc import abstractmethod
import argparse
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


class DefaultAvgPool2DModel(common.DefaultOpTestModel):
    @abstractmethod
    def build_core_model(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        self._prep_backend()
        self.build_core_model(*args, **kwargs)
        self.core_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.core_model.summary()


class AvgPool2D(DefaultAvgPool2DModel):
    def build_core_model(self, height, width, input_channels, *, pool_size,
                         strides, padding):
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        assert padding.lower() == "valid", "padding mode must be valid"
        assert (height % 2 == 1 or width % 2 == 1 or pool_size[0] != 2
                or pool_size[1] != 2 or strides[0] != 2 or strides[1] != 2
                ), "parameters must differ from what avgpool2d_2x2 can match"

        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.AveragePooling2D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    input_shape=(height, width, input_channels),
                )
            ],
        )


def main(path=DEFAULT_PATH,
         *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         pool_size=DEFAULT_POOL_SIZE,
         strides=DEFAULT_STRIDES,
         padding=DEFAULT_PADDING):
    model = AvgPool2D("avgpool2d", Path(path))
    model.build(
        height,
        width,
        input_channels,
        padding=padding,
        pool_size=pool_size,
        strides=strides,
    )
    model.run()


if __name__ == "__main__":
    parser = common.OpTestPoolStridesParser(
        defaults={
            "path": DEFAULT_PATH,
            "inputs": DEFAULT_INPUTS,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
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
