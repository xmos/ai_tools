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
import tflite2xcore_graph_conv as graph_conv

from tensorflow import keras
from copy import deepcopy
from tflite_utils import load_tflite_as_json
from tflite2xcore_utils import clean_unused_buffers, clean_unused_opcodes, clean_unused_tensors


DIRNAME = pathlib.Path(__file__).parent
MODELS_DIR, DATA_DIR = utils.make_aux_dirs(DIRNAME)


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
            'x_test': np.float32(x_test), 'y_test' : np.float32(y_test)}


def build_model(input_dim, out_dim=2):
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(input_dim, 1, 1)),
        keras.layers.Dense(out_dim, activation='softmax')
    ])


def add_example_data(model, data, classes):
    # get input quantization
    subgraph = model['subgraphs'][0]
    input_tensor = subgraph['tensors'][subgraph['inputs'][0]]
    quantization = input_tensor['quantization']

    # prepare example test data
    subset_inds = np.searchsorted(data['y_test'].flatten(), np.arange(classes))
    x_test = utils.quantize_data(data['x_test'][subset_inds],
        quantization['scale'], quantization['zero_point'])  # pylint: disable=unsubscriptable-object
    y_test = utils.one_hot_encode(data['y_test'][subset_inds], classes)  # pylint: disable=unsubscriptable-object

    # add test data to xcore_model
    subgraph['tensors'].append({
        'shape': list(x_test.shape),
        'type': 'INT8',
        'buffer': len(model['buffers']),
        'name': 'x_test',
        'is_variable': False,
        'quantization': quantization
    })
    model['buffers'].append({'data': list(x_test.flatten().tostring())})

    subgraph['tensors'].append({
        'shape': list(y_test.shape),
        'type': 'INT8',
        'buffer': len(model['buffers']),
        'name': 'y_test',
        'is_variable': False,
        'quantization': quantization
    })
    model['buffers'].append({'data': list(y_test.flatten().tostring())})


def main(input_dim=32, classes=2, *, train_new_model=True):
    assert input_dim % 32 == 0
    assert 1 < classes < 16

    data_path = DATA_DIR / 'dataset_float.npz'
    keras_model_path = MODELS_DIR / "model.h5"

    if train_new_model:
        keras.backend.clear_session()
        utils.set_all_seeds()

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
        logging.info(f"Loading data from {data_path}")
        data = dict(np.load(data_path))
        logging.info(f"Loading keras from {keras_model_path}")
        model = keras.models.load_model(keras_model_path)
        assert model.output_shape[1]==classes

    # convert to TFLite float, save and visualize
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    utils.save_from_tflite_converter(converter, MODELS_DIR, "model_float")

    # convert to TFLite quantized
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    x_train_ds = tf.data.Dataset.from_tensor_slices(data['x_train']).batch(1)
    def representative_data_gen():
        for input_value in x_train_ds.take(data['x_train'].shape[0]):  # pylint: disable=unsubscriptable-object
            yield [input_value]
    converter.representative_dataset = representative_data_gen
    model_quant_file = utils.save_from_tflite_converter(converter, MODELS_DIR, "model_quant")

    # convert quantized model to xCORE optimized
    model_quant = load_tflite_as_json(model_quant_file)
    graph_conv.remove_float_inputs_outputs(model_quant)
    graph_conv.remove_output_softmax(model_quant)

    for base_file_name in ['model_xcore', 'model_stripped']:
        model_json = deepcopy(model_quant)
        if base_file_name is 'model_xcore':
            graph_conv.replace_ops_with_XC(model_json)

        clean_unused_opcodes(model_json)
        clean_unused_tensors(model_json)
        clean_unused_buffers(model_json)

        add_example_data(model_json, data, classes)

        utils.save_from_json(model_json, MODELS_DIR, base_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu',  action='store_true', default=False,
                        help='Use GPU for training. Might result in non-reproducible results')
    parser.add_argument('--classes', type=int, default=4,
                        help='Number of classes, must be between 2 and 15.')
    parser.add_argument('--inputs', type=int, default=32,
                        help='Input dimension, must be multiple of 32.')
    parser.add_argument('--train_model',  action='store_true', default=False,
                        help='Train new model instead of loading pretrained tf.keras model.')
    parser.add_argument('-v', '--verbose',  action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    logging.info(f"Eager execution enabled: {tf.executing_eagerly()}")

    utils.set_gpu_usage(args.use_gpu, verbose)

    main(input_dim=args.inputs, classes=args.classes, train_new_model=args.train_model)
