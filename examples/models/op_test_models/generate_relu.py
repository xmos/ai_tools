#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
import logging
from tflite2xcore.model_generation import utils
from tflite2xcore.model_generation.interface import KerasModel
import tensorflow as tf
import op_test_models_common as common

DEFAULT_INPUTS = 16
DEFAULT_WIDTH = 3
DEFAULT_HEIGHT = 5
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'relu').resolve()


class ReLU(KerasModel):
    def build(self, height, width, input_channels):
        super().build()
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.ReLU(input_shape=(height, width, input_channels))
            ]
        )
        # Compilation
        self.core_model.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
        # Show summary
        self.core_model.summary()

    def train(self):  # Not training this model
        pass

    # For training
    def prep_data(self, height, width):
        self.data['export_data'], self.data['quant'] = utils.generate_dummy_data(*self.input_shape)

    # For exports
    def gen_test_data(self, height, width):
        if not self.data:
            self.prep_data(height, width)


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH):

    # Instantiate model
    test_model = ReLU('relu', Path(path))
    # Build model and compile
    test_model.build(height, width, input_channels,)
    # Generate test data
    test_model.gen_test_data(height, width)
    # Save model
    test_model.save_core_model()
    # Populate converters
    test_model.populate_converters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-path', nargs='?', default=DEFAULT_PATH,
        help='Path to a directory where models and data will be saved in subdirectories.')
    parser.add_argument(
        '-in', '--inputs', type=int, default=DEFAULT_INPUTS,
        help='Number of input channels')
    parser.add_argument(
        '-hi', '--height', type=int, default=DEFAULT_HEIGHT,
        help='Height of input image')
    parser.add_argument(
        '-wi', '--width', type=int, default=DEFAULT_WIDTH,
        help='Width of input image')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Verbose mode.')
    parser = common.parser_add_initializers(parser)
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    main(path=args.path,
         input_channels=args.inputs,
         height=args.height, width=args.width)
