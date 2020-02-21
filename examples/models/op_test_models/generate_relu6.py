#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
from tflite2xcore.model_generation.interface import FunctionModel
import op_test_models_common as common

DEFAULT_INPUTS = 16
DEFAULT_WIDTH = 3
DEFAULT_HEIGHT = 5
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'relu6').resolve()


class ReLU6(common.DefaultOpTestModel):
    def build_core_model(self, height, width, input_channels, *, input_init):
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.Lambda(
                    lambda x: tf.nn.relu6(x),
                    input_shape=(height, width, input_channels)
                )
            ]
        )
        self.input_init = input_init


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         input_init=common.DEFAULT_UNIF_INIT):

    # Instantiate model
    test_model = ReLU6('relu6', Path(path))
    test_model.build(height, width, input_channels,
                     input_init=input_init)
    test_model.gen_test_data()
    test_model.save_core_model()
    test_model.populate_converters()


if __name__ == "__main__":
    parser = common.OpTestDimParser(defaults={
        "path": DEFAULT_PATH,
        "inputs": DEFAULT_INPUTS,
        "height": DEFAULT_HEIGHT,
        "width": DEFAULT_WIDTH,
    })
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    initializers = common.initializer_args_handler(args)

    main(path=args.path,
         input_channels=args.inputs,
         height=args.height,
         width=args.width,
         input_init=initializers['input_init'])
