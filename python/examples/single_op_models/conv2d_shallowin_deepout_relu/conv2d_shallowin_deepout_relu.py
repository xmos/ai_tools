#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

# always load examples_common first to avoid debug info dump from tf initialization
import examples.examples_common as common

import os
import argparse
import logging
import pathlib
import tflite_utils

import tensorflow as tf
import numpy as np
import tflite2xcore_graph_conv as graph_conv

from copy import deepcopy
from tensorflow import keras
from tflite2xcore_utils import clean_unused_buffers, clean_unused_tensors
from tflite2xcore_utils import XCOps


DIRNAME = pathlib.Path(__file__).parent
MODELS_DIR, DATA_DIR = common.make_aux_dirs(DIRNAME)

DEFAULT_INPUTS = 3
DEFAULT_OUTPUTS = 16
DEFAULT_K_H = 3
DEFAULT_K_W = DEFAULT_K_H
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT


def build_model(inputs, outputs, K_h, K_w, height, width):
    return keras.Sequential([
        keras.layers.Conv2D(filters=outputs,
                            kernel_size=(K_h, K_w),
                            padding='same',
                            input_shape=(height, width, inputs))
    ])


def main(inputs=DEFAULT_INPUTS,
         outputs=DEFAULT_OUTPUTS,
         K_h=DEFAULT_K_H,
         K_w=DEFAULT_K_W,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH):
    keras.backend.clear_session()
    tflite_utils.set_all_seeds()

    assert inputs <= 4, "Number of input channels must be at most 4"
    assert K_w <= 8, "Kernel width must be at most 8"
    assert outputs % 16 == 0, "Number of output channels must be multiple of 16"

    # create model
    model = build_model(
        inputs=inputs, outputs=outputs, K_h=K_h, K_w=K_w, height=height, width=width)
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
    model_xcore_file = common.save_from_json(model_xcore, MODELS_DIR, 'model_xcore')
    common.save_test_data_for_xcore_model(
        model_xcore, x_test_float, data_dir=DATA_DIR, pad_input_channel_dim=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Use GPU for training. Might result in non-reproducible results')
    parser.add_argument('--inputs', type=int, default=DEFAULT_INPUTS,
                        help='Number of input channels')
    parser.add_argument('--outputs', type=int, default=DEFAULT_OUTPUTS,
                        help='Number of output channels')
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT,
                        help='Height of input image')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH,
                        help='Width of input image')
    parser.add_argument('--K_h', type=int, default=DEFAULT_K_H,
                        help='Height of kernel')
    parser.add_argument('--K_w', type=int, default=DEFAULT_K_W,
                        help='Width of kernel')
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

    main(inputs=args.inputs, outputs=args.outputs,
         K_h=args.K_h, K_w=args.K_w,
         height=args.height, width=args.width)
