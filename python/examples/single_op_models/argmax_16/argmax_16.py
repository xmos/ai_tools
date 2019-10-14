#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

# always load examples_common first to avoid debug info dump from tf initialization
import examples.examples_common as utils

import os
import argparse
import logging
import pathlib

import tensorflow as tf
import numpy as np

from tflite_utils import load_tflite_as_json
from tflite2xcore_utils import clean_unused_buffers, clean_unused_tensors
from tflite2xcore_utils import XCOps


DIRNAME = pathlib.Path(__file__).parent
MODELS_DIR, DATA_DIR = utils.make_aux_dirs(DIRNAME)

DEFAULT_INPUTS = 10


class ArgMaxModel(tf.Module):

    def __init__(self):
        pass

    @tf.function
    def func(self, x):
        return tf.math.argmax(x, axis=1, output_type=tf.int32)


def main(inputs=DEFAULT_INPUTS):
    # create model
    model = ArgMaxModel()
    concrete_func = model.func.get_concrete_function(  # pylint: disable=no-member
        tf.TensorSpec([1, inputs], tf.float32))

    # generate example data
    utils.set_all_seeds()
    x_test_float = np.float32(np.random.uniform(0, 1, size=(inputs, inputs)))
    x_test_float += np.eye(inputs)

    # convert to TFLite float, save model and visualization, save test data
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    utils.save_from_tflite_converter(converter, MODELS_DIR, "model_float")
    utils.save_test_data_for_converter(
        converter, x_test_float, data_dir=DATA_DIR, base_file_name="model_float")

    # convert to TFLite quantized, save model and visualization, save test data
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    utils.quantize_converter(converter, x_test_float)
    model_quant_file = utils.save_from_tflite_converter(converter, MODELS_DIR, "model_quant")
    utils.save_test_data_for_converter(
        converter, x_test_float, data_dir=DATA_DIR, base_file_name="model_quant")

    # load quantized model in json, serving as basis for conversions
    # strip quantized model of float interface and softmax
    model_quant = load_tflite_as_json(model_quant_file)
    model_stripped = utils.strip_model_quant(model_quant)
    model_stripped['description'] = "TOCO Converted and stripped."
    model_stripped_file = utils.save_from_json(model_stripped, MODELS_DIR, 'model_stripped')
    utils.save_test_data_for_stripped_model(
        model_stripped, x_test_float, data_dir=DATA_DIR, add_float_outputs=False)

    # load stripped model in json, converting manually
    model_xcore = load_tflite_as_json(model_stripped_file)
    subgraph = model_xcore['subgraphs'][0]

    # update operator details
    operator = subgraph['operators'][0]
    operator['builtin_options']
    del operator['builtin_options']
    operator['inputs'] = operator['inputs'][:1]

    # change type and quantization of input to INT16
    input_tensor = subgraph['tensors'][subgraph['inputs'][0]]
    input_tensor['type'] = 'INT16'
    input_tensor['name'] = 'x_int16'
    input_quant = input_tensor['quantization']
    input_quant['zero_point'] = [input_quant['zero_point'][0] * 2**8]
    input_quant['scale'] = [input_quant['scale'][0] / 2**8]

    # replace opcode to xcore version
    model_xcore['operator_codes'][0] = {'builtin_code': 'CUSTOM',
                                        'custom_code': XCOps.ARGMAX_16,
                                        'version': 1}

    clean_unused_tensors(model_xcore)
    clean_unused_buffers(model_xcore)
    utils.save_from_json(model_xcore, MODELS_DIR, 'model_xcore')
    utils.save_test_data_for_xcore_model(model_xcore, x_test_float, data_dir=DATA_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=int, default=DEFAULT_INPUTS,
                        help='Input dimension')
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
