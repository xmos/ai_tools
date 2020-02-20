#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from abc import abstractmethod
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
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


# TODO: refactor this since other single op models also use something similar
class DefaultPool2DModel(common.DefaultOpTestModel):
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


class AvgPool2D(DefaultPool2DModel):
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
                tf.keras.layers.AveragePooling2D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    input_shape=(height, width, input_channels),
                )
            ],
        )


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         pool_size=DEFAULT_POOL_SIZE,
         strides=DEFAULT_STRIDES,
         padding=DEFAULT_PADDING):
    model = AvgPool2D("avgpool2d", Path(path))
    model.build(height, width, input_channels,
                pool_size=pool_size, strides=strides, padding=padding)
    model.run()


# TODO: this should become part of the parser
def strides_pool_arg_handler(args):
    parameters = {
        "strides": (DEFAULT_STRIDE_HEIGHT, DEFAULT_STRIDE_WIDTH),
        "pool_size": (DEFAULT_POOL_HEIGHT, DEFAULT_POOL_WIDTH),
    }
    arguments = {k: vars(args)[k] for k in parameters if k in vars(args)}
    for k in arguments:
        params = arguments[k]
        if len(params) > 2:
            raise argparse.ArgumentTypeError(
                f"The {k} argument must be at most 2 numbers."
            )
        else:
            arguments[k] = tuple(params) if len(params) == 2 else (params[0],) * 2

    return arguments


if __name__ == "__main__":
    parser = common.OpTestDimParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'height': DEFAULT_HEIGHT,
        'width': DEFAULT_WIDTH,
        'padding': DEFAULT_PADDING,
    })
    parser.add_argument(
        "-st", "--strides", nargs="+", type=int, default=argparse.SUPPRESS,
        help="Strides, vertical first "
        f"(default: {DEFAULT_STRIDE_HEIGHT} {DEFAULT_STRIDE_WIDTH})",
    )
    parser.add_argument(
        '-po', '--pool_size', nargs='+', type=int, default=argparse.SUPPRESS,
        help="Pool size:, vertical first "
        f"(default: {DEFAULT_POOL_HEIGHT} {DEFAULT_POOL_WIDTH})",
    )
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    strides_pool = strides_pool_arg_handler(args)

    main(path=args.path,
         input_channels=args.inputs,
         height=args.height,
         width=args.width,
         padding=args.padding,
         **strides_pool)
