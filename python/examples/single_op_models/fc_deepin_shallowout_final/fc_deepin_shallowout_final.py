#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

# always load examples_common first to avoid debug info dump from tf initialization
from examples.examples_common import set_all_seeds, make_aux_dirs
from examples.examples_common import quantize_data, one_hot_encode

import os
import argparse
import logging
import pathlib
import tflite2xcore_graph_conv

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tflite_utils import load_tflite_as_json


DIRNAME = pathlib.Path(__file__).parent
MODELS_DIR, DATA_DIR = make_aux_dirs(DIRNAME)


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


def main(input_dim=32, classes=2):
    assert input_dim % 32 == 0
    assert 1 < classes < 16

    keras.backend.clear_session()
    set_all_seeds()

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
    np.savez(DATA_DIR / 'dataset_float', **data)
    model.save(os.path.join(MODELS_DIR / "model.h5"))

    # convert to TFLite float
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_float_lite = converter.convert()

    model_float_file = MODELS_DIR / "model_float.tflite"
    size_float = model_float_file.write_bytes(model_float_lite)
    logging.info('Float model size: {:.0f} KB'.format(size_float/1024))

    # convert to TFLite quantized
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    x_train_ds = tf.data.Dataset.from_tensor_slices(data['x_train']).batch(1)
    def representative_data_gen():
        for input_value in x_train_ds.take(100):
            yield [input_value]
    converter.representative_dataset = representative_data_gen
    model_quant_lite = converter.convert()

    model_quant_file = MODELS_DIR / "model_quant.tflite"
    size_quant = model_quant_file.write_bytes(model_quant_lite)
    logging.info('Quantized model size: {:.0f} KB'.format(size_quant/1024))

    # convert quantized model to xCORE optimized
    model_xcore_file = MODELS_DIR / "model_quant_xcore.tflite"
    tflite2xcore_graph_conv.main(model_quant_file, model_xcore_file,
         remove_softmax=True)

    # get input quantization
    model_xcore = load_tflite_as_json(model_xcore_file)
    subgraph = model_xcore['subgraphs'][0]
    input_tensor = subgraph['tensors'][subgraph['inputs'][0]]
    scale = input_tensor['quantization']['scale']
    zero_point = input_tensor['quantization']['zero_point']

    # quantize data
    data['x_train'] = quantize_data(data['x_train'], scale, zero_point)
    data['x_test'] = quantize_data(data['x_test'], scale, zero_point)

    # one-hot encode labels
    data['y_train'] = one_hot_encode(data['y_train'], classes)
    data['y_test'] = one_hot_encode(data['y_test'], classes)

    # save quantized data
    np.savez(DATA_DIR / 'dataset_quant', **data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu',  action='store_true', default=False,
                        help='Use GPU for training. Might result in non-reproducible results')
    parser.add_argument('-v', '--verbose',  action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    logging.info(f"Eager execution enabled: {tf.executing_eagerly()}")

    # can throw annoying error if CUDA cannot be initialized
    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    if gpus:
        if args.use_gpu:
            tf.config.experimental.set_memory_growth(gpus[0], enable=True)
        else:
            if verbose:
                logging.info("GPUs disabled.")
            tf.config.experimental.set_visible_devices([], 'GPU')
    elif args.use_gpu:
        logging.warning('No available GPUs found, defaulting to CPU.')

    main(classes=10)
