# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import shutil
import pathlib
import argparse
import logging
import random
import numpy as np
import tensorflow as tf
from termcolor import colored

import model_interface as mi
import tflite_utils


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


def printc(*s, c='green', back='on_grey'):
    if len(s) == 1:
        print(colored(str(s)[2:-3], c, back))
    else:
        print(colored(s[0], c, back), str(s[1:])[1:-2])


def debug_dir(path, name, before):
    if before:
        printc(name + ' directory before generation:')
    else:
        printc(name + ' directory after generation:')
    print([str(x.name) for x in path.iterdir() if x.is_file() or x.is_dir()])


def debug_keys_header(title, test_model):
    printc(title, c='blue')
    debug_keys('Model keys:\n', test_model.models)
    debug_keys('Data keys:\n', test_model.data)
    debug_keys('Converter keys:\n', test_model.converters)


def debug_keys(string, dic):
    printc(string, dic.keys())


def debug_conv(to_type, test_model, datapath, modelpath):
    debug_keys_header('Conversion to ' + to_type + ' start', test_model)
    debug_dir(modelpath, 'Models', True)
    printc('Converting model...', c='yellow')
    choose_conv_or_save(to_type, test_model, False)
    debug_dir(modelpath, 'Models', False)
    debug_dir(datapath, 'Data', True)
    printc('Saving data...', c='yellow')
    choose_conv_or_save(to_type, test_model, True)
    debug_dir(datapath, 'Data', False)


def choose_conv_or_save(conv, test_model, save):
    if not save:
        return{
            'float': lambda m: m.to_tf_float(),
            'quant': lambda m: m.to_tf_quant(),
            'stripped': lambda m: m.to_tf_stripped(),
            'xcore': lambda m: m.to_tf_xcore()
        }[conv](test_model)
    else:
        return{
            'float': lambda m: m.save_tf_float_data(),
            'quant': lambda m: m.save_tf_quant_data(),
            'stripped': lambda m: m.save_tf_stripped_data(add_float_outputs=False),
            'xcore': lambda m: m.save_tf_xcore_data()
        }[conv](test_model)


def main():
    DEFAULT_INPUTS = 10
    # Random seed
    random.seed(42)
    # Remove everthing
    if os.path.exists('./debug/function_test'):
        shutil.rmtree('./debug/function_test')
    modelpath = pathlib.Path('./debug/function_test/models')
    datapath = pathlib.Path('./debug/function_test/test_data')
    # Instantiation
    test_model = ArgMax16(
        'arg_max_16', pathlib.Path('./debug/function_test'), DEFAULT_INPUTS)
    printc('Model name property:\n', test_model.name)
    # Build
    debug_keys_header('Keys before build()', test_model)
    debug_dir(modelpath, 'Models', True)
    debug_dir(datapath, 'Data', True)
    test_model.build()
    '''
    # Train data preparation
    test_model.prep_data()
    debug_keys_header('Keys after build() and prep_data()', test_model)
    debug_dir(modelpath, 'Models', False)
    debug_dir(datapath, 'Data', False)
    # Training
    printc('Training:', c='blue')
    test_model.train()
    '''
    # Save model
    printc('Saving model', c='blue')
    test_model.save_core_model()
    debug_dir(modelpath, 'Models', False)
    debug_dir(datapath, 'Data', False)
    # Export data generation
    printc('Generating export data', c='blue')
    test_model.gen_test_data()
    debug_keys('Data keys after export data generation:\n', test_model.data)
    # Conversions
    debug_conv('float', test_model, datapath, modelpath)
    debug_conv('quant', test_model, datapath, modelpath)
    debug_conv('stripped', test_model, datapath, modelpath)
    debug_conv('xcore', test_model, datapath, modelpath)
    # Final status
    debug_keys_header('Final status', test_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()

    verbose = args.verbose
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.info(f"Eager execution enabled: {tf.executing_eagerly()}")

    main()