# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
import model_interface as mi
import tflite_utils

DEFAULT_INPUTS = 32
DEFAULT_HEIGHT = 4
DEFAULT_WIDTH = DEFAULT_HEIGHT

DEFAULT_POOL_SIZE = 2
DEFAULT_PADDING = 'valid'
DEFAULT_STRIDES = 2
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'avgpool_2d_deep').resolve()

def generate_data(height, width, inputs):
    quant_data = np.float32(
        np.random.uniform(0, 1, size=(10, height, width, inputs)))
    x_test_float = np.concatenate(
        [np.zeros((1, height, width, inputs), dtype=np.float32),
         quant_data[:3, :, :, :]],  # pylint: disable=unsubscriptable-object
        axis=0)
    return x_test_float, quant_data

class AvgPool2d(mi.KerasModel):
    
    def build(self, height, width, pool, stride, pad):
        # Env, TODO: consider refactoring this to KerasModel
        tf.keras.backend.clear_session()
        tflite_utils.set_all_seeds()
        # Building
        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.AveragePooling2D(
                    pool_size=pool,
                    strides=stride,
                    padding=pad,
                    input_shape=(height, width, self.input_dim))
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
        self.data['export_data'], self.data['quant'] = generate_data(
            height, width, self.input_dim)
        
    # For exports
    def gen_test_data(self, height, width):
        if not self.data:
            self.prep_data(height, width)



def main(path=DEFAULT_PATH, *,
        input_dim = DEFAULT_INPUTS,
        height = DEFAULT_HEIGHT, width = DEFAULT_WIDTH,
        pool_size = DEFAULT_POOL_SIZE,
        padding = DEFAULT_PADDING,
        strides = DEFAULT_STRIDES):
    # nstantiate model
    test_model = AvgPool2d(
        'avgpool2d_deep', path, input_dim)
    # Build model and compile
    test_model.build(height, width, pool_size, strides, padding)
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
        '-in', '--inputs', type=int, default=DEFAULT_INPUTS,
        help='Number of input channels')
    parser.add_argument(
        '-hi', '--height', type=int, default=DEFAULT_HEIGHT,
        help='Height of input image')
    parser.add_argument(
        '-wi', '--width', type=int, default=DEFAULT_WIDTH,
        help='Width of input image')
    parser.add_argument(
        '-st', '--strides', type=int, default=DEFAULT_STRIDES,
        help='Height of kernel')
    parser.add_argument(
        '-po', '--pool_size', type=int, default=DEFAULT_POOL_SIZE,
        help='Height of kernel')
    parser.add_argument(
        '-pd', '--padding', type=str, default=DEFAULT_PADDING,
        help='Width of kernel')
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
    # tflite_utils.set_gpu_usage(args.use_gpu, verbose)

    main(path=args.path,
         input_dim=args.inputs,
         height=args.height, width=args.width,
         pool_size=args.pool_size, padding=args.padding,
         strides = args.strides
         )
