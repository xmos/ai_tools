# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
import model_interface as mi
import tflite_utils

from generate_conv2d_deepin_deepout_relu import generate_data  # TODO: factor out

DEFAULT_INPUTS = 3
DEFAULT_OUTPUTS = 16
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = 'same'
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'conv2d_shallowin_deepout_relu').resolve()


class Conv2dShallowinDeepoutRelu(mi.KerasModel):
    def build(self, K_h, K_w, height, width, input_channels, output_channels, padding):
        assert input_channels <= 4, "Number of input channels must be at most 4"
        assert K_w <= 8, "Kernel width must be at most 8"
        assert output_channels % 16 == 0, "Number of output channels must be multiple of 16"
        assert K_h % 2 == 1, "kernel height must be odd"
        assert K_w % 2 == 1, "kernel width must be odd"
        super().build()

        # Building
        try:
            self.core_model = tf.keras.Sequential(
                name=self.name,
                layers=[
                    tf.keras.layers.Conv2D(filters=output_channels,
                                           kernel_size=(K_h, K_w),
                                           padding=padding,
                                           input_shape=(height, width, input_channels))
                ]
            )
        except ValueError as e:
            if e.args[0].startswith("Negative dimension size caused by"):
                raise ValueError(
                    "Negative dimension size (Hint: if using 'valid' padding "
                    "verify that the kernel is at least the size of input image)"
                ) from e

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
        self.data['export_data'], self.data['quant'] = generate_data(*self.input_shape)

    # For exports
    def gen_test_data(self, height, width):
        if not self.data:
            self.prep_data(height, width)


def main(path=DEFAULT_PATH, *,
         input_channels=DEFAULT_INPUTS, output_channels=DEFAULT_OUTPUTS,
         height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH,
         K_h=DEFAULT_KERNEL_HEIGHT, K_w=DEFAULT_KERNEL_WIDTH,
         padding=DEFAULT_PADDING):
    # Instantiate model
    test_model = Conv2dShallowinDeepoutRelu('conv2d_shallowin_deepout_relu', Path(path))
    # Build model and compile
    test_model.build(K_h, K_w, height, width, input_channels, output_channels, padding)
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
        'path', nargs='?', default=DEFAULT_PATH,
        help='Path to a directory where models and data will be saved in subdirectories.')
    parser.add_argument(
        '--use_gpu', action='store_true', default=False,
        help='Use GPU for training. Might result in non-reproducible results')
    parser.add_argument(
        '-in', '--inputs', type=int, default=DEFAULT_INPUTS,
        help='Number of input channels')
    parser.add_argument(
        '-out', '--outputs', type=int, default=DEFAULT_OUTPUTS,
        help='Number of output channels')
    parser.add_argument(
        '-hi', '--height', type=int, default=DEFAULT_HEIGHT,
        help='Height of input image')
    parser.add_argument(
        '-wi', '--width', type=int, default=DEFAULT_WIDTH,
        help='Width of input image')
    parser.add_argument(
        '-kh', '--kernel_height', type=int, default=DEFAULT_KERNEL_HEIGHT,
        help='Height of kernel')
    parser.add_argument(
        '-kw', '--kernel_width', type=int, default=DEFAULT_KERNEL_WIDTH,
        help='Width of kernel')
    parser.add_argument(
        '-pd', '--padding', type=str, default=DEFAULT_PADDING,
        help='Padding mode')
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='Verbose mode.')
    args = parser.parse_args()

    # TODO: consider refactoring this to utils
    verbose = args.verbose
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.info(f"Eager execution enabled: {tf.executing_eagerly()}")
    tflite_utils.set_gpu_usage(args.use_gpu, verbose)

    main(path=args.path,
         input_channels=args.inputs, output_channels=args.outputs,
         K_h=args.kernel_height, K_w=args.kernel_width,
         height=args.height, width=args.width,
         padding=args.padding)
