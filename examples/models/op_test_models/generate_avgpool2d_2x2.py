#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
import tensorflow as tf

from generate_avgpool2d import (
    DEFAULT_INPUTS,
    DefaultAvgPool2DModel,
    DefaultAvgPool2DParser
)

DEFAULT_HEIGHT = 4
DEFAULT_WIDTH = DEFAULT_HEIGHT

DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'avgpool2d_2x2').resolve()


class AvgPool2D2x2(DefaultAvgPool2DModel):
    def build_core_model(self, height, width, input_channels):
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        assert height % 2 == 0, "height must be even"
        assert width % 2 == 0, "width must be even"

        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.AveragePooling2D(
                    pool_size=2,
                    strides=2,
                    padding='valid',
                    input_shape=(height, width, input_channels))
            ]
        )


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH):
    model = AvgPool2D2x2('avgpool2d_2x2', Path(path))
    model.build(height, width, input_channels)
    model.run()


if __name__ == "__main__":
    parser = DefaultAvgPool2DParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'height': DEFAULT_HEIGHT,
        'width': DEFAULT_WIDTH
    })
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    main(path=args.path,
         input_channels=args.inputs,
         height=args.height, width=args.width
         )
