#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

# always load examples_common first to avoid debug info dump from tf initialization
import examples.examples_common as common

import os
import argparse
import logging
import pathlib

import tensorflow as tf
import numpy as np
import tflite2xcore_graph_conv as graph_conv
import tflite_utils

from copy import deepcopy
from tensorflow import keras
from tflite2xcore_utils import clean_unused_buffers, clean_unused_tensors
from tflite2xcore_utils import XCOps


DIRNAME = pathlib.Path(__file__).parent
MODELS_DIR, DATA_DIR = common.make_aux_dirs(DIRNAME)

DEFAULT_INPUTS = 32
DEFAULT_HEIGHT = 4
DEFAULT_WIDTH = DEFAULT_HEIGHT


def build_model(inputs, height, width):
    return keras.Sequential([
        keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid', input_shape=(height, width, inputs))
    ])


def main(inputs=DEFAULT_INPUTS, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH):
    assert inputs % 32 == 0, "Number of input channels must be multiple of 32"

    keras.backend.clear_session()
    tflite_utils.set_all_seeds()

    # create model
    model = build_model(inputs=DEFAULT_INPUTS, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # create dataset for quantization and examples
    quant_data = np.float32(np.random.uniform(0, 1, size=(10, height, width, inputs)))
    x_test_float = np.concatenate([np.zeros((1, height, width, inputs), dtype=np.float32),
                                   quant_data[:3, :, :, :]],  # pylint: disable=unsubscriptable-object
                                  axis=0)

    # convert to TFLite float, save model and visualization, save test data
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_float_file = common.save_from_tflite_converter(converter, MODELS_DIR, "model_float")
    common.save_test_data_for_regular_model(
        model_float_file, x_test_float, data_dir=DATA_DIR, base_file_name="model_float")

    # convert to TFLite quantized, save model and visualization, save test data
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    common.quantize_converter(converter, quant_data)
    model_quant_file = common.save_from_tflite_converter(converter, MODELS_DIR, "model_quant")
    common.save_test_data_for_regular_model(
        model_quant_file, x_test_float, data_dir=DATA_DIR, base_file_name="model_quant")

    # load quantized model in json, serving as basis for conversions
    # strip quantized model of float interface and softmax
    model_quant = tflite_utils.load_tflite_as_json(model_quant_file)
    model_stripped = common.strip_model_quant(model_quant)
    model_stripped['description'] = "TOCO Converted and stripped."
    common.save_from_json(model_stripped, MODELS_DIR, 'model_stripped')
    common.save_test_data_for_stripped_model(
        model_stripped, x_test_float, data_dir=DATA_DIR)

    # save xcore converted model
    model_xcore = deepcopy(model_quant)
    graph_conv.convert_model(model_xcore, remove_softmax=True)
    common.save_from_json(model_xcore, MODELS_DIR, 'model_xcore')
    common.save_test_data_for_xcore_model(
        model_xcore, x_test_float, data_dir=DATA_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Use GPU for training. Might result in non-reproducible results')
    parser.add_argument('--inputs', type=int, default=DEFAULT_INPUTS,
                        help='Number of input channels')
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT,
                        help='Height of input image')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH,
                        help='Width of input image')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    logging.info(f"Eager execution enabled: {tf.executing_eagerly()}")

    tflite_utils.set_gpu_usage(args.use_gpu, verbose)

    main(inputs=args.inputs, height=args.height, width=args.width)
