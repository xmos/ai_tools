#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

# always load examples_common first to avoid debug info dump from tf initialization
import examples.examples_common as common

import argparse
import logging
import pathlib
import tflite_utils

import tensorflow as tf
import numpy as np
import tflite2xcore_graph_conv as graph_conv

from tensorflow import keras
from copy import deepcopy


DIRNAME = pathlib.Path(__file__).parent
MODELS_DIR, DATA_DIR = common.make_aux_dirs(DIRNAME)

DEFAULT_CLASSES = 4
DEFAULT_INPUTS = 32


def generate_fake_lin_sep_dataset(classes=2, dim=32, *,
                                  train_samples_per_class=5120,
                                  test_samples_per_class=1024):
    z = np.linspace(0, 2*np.pi, dim)

    # generate data and class labels
    x_train, x_test, y_train, y_test = [], [], [], []
    for j in range(classes):
        mean = np.sin(z) + 10*j/classes
        cov = 10 * np.diag(.5*np.cos(j * z) + 2) / (classes-1)
        x_train.append(
            np.random.multivariate_normal(mean, cov, size=train_samples_per_class))
        x_test.append(
            np.random.multivariate_normal(mean, cov, size=test_samples_per_class))
        y_train.append(j * np.ones((train_samples_per_class, 1)))
        y_test.append(j * np.ones((test_samples_per_class, 1)))

    # stack arrays
    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train)
    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)

    # normalize
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # expand dimensions for TFLite compatibility
    def expand_array(arr):
        return np.reshape(arr, arr.shape + (1, 1))
    x_train = expand_array(x_train)
    x_test = expand_array(x_test)

    return {'x_train': np.float32(x_train), 'y_train': np.float32(y_train),
            'x_test': np.float32(x_test), 'y_test': np.float32(y_test)}


def build_model(input_dim, out_dim=2):
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(input_dim, 1, 1)),
        keras.layers.Dense(out_dim, activation='softmax')
    ])


def main(input_dim=DEFAULT_INPUTS, classes=DEFAULT_CLASSES, *, train_new_model=False):
    assert input_dim % 32 == 0
    assert 1 < classes < 16

    data_path = DATA_DIR / 'training_dataset.npz'
    keras_model_path = MODELS_DIR / "model.h5"

    if train_new_model:
        keras.backend.clear_session()
        tflite_utils.set_all_seeds()

        # create model
        model = build_model(input_dim, out_dim=classes)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        # generate data
        data = generate_fake_lin_sep_dataset(classes, input_dim,
                                             train_samples_per_class=51200//classes,
                                             test_samples_per_class=10240//classes)

        # run the training
        model.fit(data['x_train'], data['y_train'],
                  epochs=5*(classes-1), batch_size=128,
                  validation_data=(data['x_test'], data['y_test']))

        # save model and data
        np.savez(data_path, **data)
        model.save(keras_model_path)

    else:
        try:
            logging.info(f"Loading data from {data_path}")
            data = dict(np.load(data_path))
            logging.info(f"Loading keras from {keras_model_path}")
            model = keras.models.load_model(keras_model_path)
        except FileNotFoundError as e:
            logging.error(f"{e} (Hint: use the --train_model flag)")
            return
        if model.output_shape[1] != classes:
            raise ValueError(f"number of specified classes ({classes}) "
                             f"does not match model output shape ({model.output_shape[1]})")

    # choose test data examples
    subset_inds = np.searchsorted(data['y_test'].flatten(), np.arange(classes))
    x_test_float = data['x_test'][subset_inds]

    # convert to TFLite float, save model and visualization, save test data
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_float_file = common.save_from_tflite_converter(converter, MODELS_DIR, "model_float")
    common.save_test_data_for_regular_model(
        model_float_file, x_test_float, data_dir=DATA_DIR, base_file_name="model_float")

    # convert to TFLite quantized, save model and visualization, save test data
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    common.quantize_converter(converter, data['x_train'])
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
    parser.add_argument('--classes', type=int, default=DEFAULT_CLASSES,
                        help='Number of classes, must be between 2 and 15.')
    parser.add_argument('--inputs', type=int, default=DEFAULT_INPUTS,
                        help='Input dimension, must be multiple of 32.')
    parser.add_argument('--train_model', action='store_true', default=False,
                        help='Train new model instead of loading pretrained tf.keras model.')
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

    main(input_dim=args.inputs, classes=args.classes, train_new_model=args.train_model)
