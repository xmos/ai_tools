#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

# always load examples_common first to avoid debug info dump from tf initialization
import examples.examples_common as utils

import os
import argparse
import logging
import pathlib
import tempfile

import tensorflow as tf
import numpy as np
import tflite2xcore_graph_conv as graph_conv

from tensorflow import keras
from copy import deepcopy
from tflite_utils import load_tflite_as_json


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


def apply_interpreter_to_examples(interpreter, examples):
    interpreter_input_ind = interpreter.get_input_details()[0]["index"]
    interpreter_output_ind = interpreter.get_output_details()[0]["index"]
    interpreter.allocate_tensors()

    outputs = []
    for x in examples:
        interpreter.set_tensor(interpreter_input_ind, tf.expand_dims(x, 0))
        interpreter.invoke()
        y = interpreter.get_tensor(interpreter_output_ind)
        outputs.append(y)

    return outputs


def save_test_data(data, base_file_name):
    # save test data in numpy format
    test_data_dir = DATA_DIR / base_file_name
    test_data_dir.mkdir(exist_ok=True, parents=True)
    np.savez(test_data_dir / f"{base_file_name}.npz", **data)

    # save individual binary files for easier low level access
    for key, test_set in data.items():
        for j, arr in enumerate(test_set):
            with open(test_data_dir / f"test_{j}.{key[0]}", 'wb') as f:
                f.write(arr.flatten().tostring())

    logging.info(f"test examples for {base_file_name} saved to {test_data_dir}")


def save_test_data_for_converter(converter, x_test_float, *, base_file_name):
    # create interpreter
    interpreter = tf.lite.Interpreter(model_content=converter.convert())

    # extract reference labels for the test examples
    logging.info(f"Extracting examples for {base_file_name}...")
    y_test = apply_interpreter_to_examples(interpreter, x_test_float)
    data = {'x_test': x_test_float, 'y_test': np.vstack(y_test)}

    # save data
    save_test_data(data, base_file_name)


def save_test_data_for_stripped_model(model_stripped, x_test_float, *,
                                       base_file_name='model_stripped'):
    model_stripped = deepcopy(model_stripped)

    # extract quantization info of input/output
    subgraph = model_stripped['subgraphs'][0]
    input_tensor = subgraph['tensors'][subgraph['inputs'][0]]
    output_tensor = subgraph['tensors'][subgraph['outputs'][0]]
    input_quant = input_tensor['quantization']
    output_quant = output_tensor['quantization']

    # add float interface
    graph_conv.add_float_inputs_outputs(model_stripped)

    # the TFLite interpreter needs a temporary file
    # the lifetime of this file needs to be at least the lifetime of the interpreter
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_tmp_file = utils.save_from_json(
            model_stripped, pathlib.Path(tmp_dir), 'model_tmp', visualize=False)

        # create interpreter
        interpreter = tf.lite.Interpreter(model_path=str(model_tmp_file))

        # quantize test examples
        x_test = utils.quantize(
            x_test_float, input_quant['scale'][0], input_quant['zero_point'][0])

        # extract and quantize reference labels for the test examples
        logging.info(f"Extracting examples for {base_file_name}...")
        y_test = apply_interpreter_to_examples(interpreter, x_test_float)
        y_test = map(
            lambda y: utils.quantize(y, output_quant['scale'][0], output_quant['zero_point'][0]),
            y_test
        )
        data = {'x_test': x_test, 'y_test': np.vstack(list(y_test))}

    # save data
    save_test_data(data, base_file_name)


def save_test_data_for_xcore_model(model_xcore, x_test_float, *,
                                   base_file_name='model_xcore'):

    # extract quantization info of input/output
    subgraph = model_xcore['subgraphs'][0]
    input_tensor = subgraph['tensors'][subgraph['inputs'][0]]
    input_quant = input_tensor['quantization']

    # quantize test examples
    x_test = utils.quantize(
        x_test_float, input_quant['scale'][0], input_quant['zero_point'][0])
    
    # save data
    save_test_data({'x_test': x_test}, base_file_name)


def main(input_dim=32, classes=2, *, train_new_model=True):
    assert input_dim % 32 == 0
    assert 1 < classes < 16

    data_path = DATA_DIR / 'training_dataset.npz'
    keras_model_path = MODELS_DIR / "model.h5"

    if train_new_model:
        keras.backend.clear_session()
        utils.set_all_seeds()

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
    utils.save_from_tflite_converter(converter, MODELS_DIR, "model_float")

    save_test_data_for_converter(converter, x_test_float, base_file_name="model_float")

    # convert to TFLite quantized, save model and visualization, save test data
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    x_train_ds = tf.data.Dataset.from_tensor_slices(data['x_train']).batch(1)
    def representative_data_gen():
        for input_value in x_train_ds.take(data['x_train'].shape[0]):  # pylint: disable=unsubscriptable-object
            yield [input_value]
    converter.representative_dataset = representative_data_gen
    model_quant_file = utils.save_from_tflite_converter(converter, MODELS_DIR, "model_quant")
    save_test_data_for_converter(converter, x_test_float, base_file_name="model_quant")

    # load quantized model in json, serving as basis for conversions
    # strip quantized model of float interface and softmax
    model_quant = load_tflite_as_json(model_quant_file)
    model_stripped = utils.strip_model_quant(model_quant)
    model_stripped['description'] = "TOCO Converted and stripped."
    utils.save_from_json(model_stripped, MODELS_DIR, 'model_stripped')
    save_test_data_for_stripped_model(model_stripped, x_test_float)

    # save xcore converted model
    model_xcore = deepcopy(model_quant)
    graph_conv.convert_model(model_xcore, remove_softmax=True)
    model_xcore_file = utils.save_from_json(model_xcore, MODELS_DIR, 'model_xcore')
    save_test_data_for_xcore_model(model_xcore, x_test_float)

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
