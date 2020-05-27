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

DEFAULT_PATH = Path(__file__).parent.joinpath("debug", "maxpool2d").resolve()
DEFAULT_NUM_THREADS = 1


class MaxPool2d(common.OpTestDefaultModel):
    def build_core_model(
        self, height, width, input_channels, *, pool_size, strides, padding, input_init
    ):
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        if padding.lower() == "same":
            assert (
                height % 2 == width % 2 == 0
                and pool_size[0] == pool_size[1] == 2
                and strides[0] == strides[1] == 2
            ), "same padding is only allowed for the common 2x2 case"
        else:
            assert padding.lower() == "valid", f"invalid padding mode '{padding}'"
        self.input_init = input_init
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


def main(raw_args=None):
    parser = common.OpTestPoolParser(
        defaults={
            "path": DEFAULT_PATH,
            "inputs": DEFAULT_INPUTS,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "padding": DEFAULT_PADDING,
            "strides": DEFAULT_STRIDES,
            "pool_size": DEFAULT_POOL_SIZE,
            "inits": {"input_init": {"type": common.OpTestInitializers.UNIF}},
        }
    )
    parser.add_argument(
        "-par",
        "--num_threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help="Number of parallel threads for xcore.ai optimization.",
    )
    args = parser.parse_args(raw_args)

    model = MaxPool2d("maxpool2d", args.path)
    model.build(
        args.height,
        args.width,
        args.inputs,
        padding=args.padding,
        strides=args.strides,
        pool_size=args.pool_size,
        **args.inits,
    )
    model.run(num_threads=args.num_threads)


if __name__ == "__main__":
    main()
