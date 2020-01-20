# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
import model_interface as mi
import tflite_utils


DEFAULT_INPUTS = 10
DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'arg_max_16').resolve()


class ArgMax16(mi.FunctionModel):
    def build(self):
        class ArgMaxModel(tf.Module):

            def __init__(self):
                pass

            @tf.function
            def func(self, x):
                return tf.math.argmax(x, axis=1, output_type=tf.int32)
        self.core_model = ArgMaxModel()

    @property
    def function_model(self):
        return [self.core_model.func.get_concrete_function(
                tf.TensorSpec([1, self.input_dim], tf.float32))]

    def prep_data(self):  # Not training this model
        pass

    def train(self):  # Not training this model
        pass

    def gen_test_data(self):
        tflite_utils.set_all_seeds()
        x_test_float = np.float32(
            np.random.uniform(0, 1, size=(self.input_dim, self.input_dim)))
        x_test_float += np.eye(self.input_dim)
        self.data['export_data'] = x_test_float
        self.data['quant'] = x_test_float


def main(path=DEFAULT_PATH, inputs=DEFAULT_INPUTS):
    # Instantiate model
    test_model = ArgMax16(
        'arg_max_16', path, DEFAULT_INPUTS)

    # Build model
    test_model.build()

    # Save model
    test_model.save_core_model()
    # test_model.save_core_model() - breaks on conversions
    
    # Export data generation
    test_model.gen_test_data()

    # Populate converters and data
    test_model.populate_converters(add_float_outputs=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=int, default=DEFAULT_INPUTS,
                        help='Input dimension')
    parser.add_argument(
        'path', nargs='?', default=DEFAULT_PATH,
        help='Path to a directory where models and data will be saved in subdirectories.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    logging.info(f"Eager execution enabled: {tf.executing_eagerly()}")

    main(inputs=args.inputs)