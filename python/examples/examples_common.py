# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import random
import logging
import pathlib
import tempfile

import numpy as np

from copy import deepcopy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf  # nopep8
warnings.filterwarnings(action='default')

import tflite_visualize
from tflite_utils import save_json_as_tflite
import tflite2xcore_utils
import tflite2xcore_graph_conv as graph_conv


DEFAULT_SEED = 123


def set_all_seeds(seed=DEFAULT_SEED):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_aux_dirs(dirname):
    models_dir = dirname / "models"
    data_dir = dirname / "test_data"
    models_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)

    return models_dir, data_dir


def load_scaled_cifar10():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    scale = tf.constant(255, dtype=tf.dtypes.float32)
    x_train, x_test = train_images/scale - .5, test_images/scale - .5
    y_train, y_test = train_labels, test_labels

    return (x_train, y_train), (x_test, y_test)


def quantize(arr, scale, zero_point, dtype=np.int8):
    t = np.round(arr / scale + zero_point)
    return dtype(np.round(np.clip(t, np.iinfo(dtype).min, np.iinfo(dtype).max)))


def dequantize(arr, scale, zero_point):
    return (np.float32(arr) - zero_point) * scale


def one_hot_encode(arr, classes):
    return tf.keras.utils.to_categorical(arr, classes, dtype=np.int8)


def set_gpu_usage(use_gpu, verbose):
    # can throw annoying error if CUDA cannot be initialized
    default_log_level = os.environ['TF_CPP_MIN_LOG_LEVEL']
    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = default_log_level

    if gpus:
        if use_gpu:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, enable=True)
        else:
            logging.info("GPUs disabled.")
            tf.config.experimental.set_visible_devices([], 'GPU')
    elif use_gpu:
        logging.warning('No available GPUs found, defaulting to CPU.')


def save_from_tflite_converter(converter, models_dir, base_file_name, *,
                               visualize=True):
    model = converter.convert()

    model_file = models_dir / f"{base_file_name}.tflite"
    model_html = models_dir / f"{base_file_name}.html"
    size = model_file.write_bytes(model)
    logging.info(f"{base_file_name} size: {size/1024:.0f} KB".format())
    if visualize:
        tflite_visualize.main(model_file, model_html)
        logging.info(f"{base_file_name} visualization saved to {os.path.realpath(model_html)}")

    return model_file


def save_from_json(model, models_dir, base_file_name, *,
                   visualize=True):
    model_file = models_dir / f"{base_file_name}.tflite"
    model_html = models_dir / f"{base_file_name}.html"
    save_json_as_tflite(model, model_file)

    if visualize:
        tflite_visualize.main(model_file, model_html)
        logging.info(f"{base_file_name} visualization saved to {os.path.realpath(model_html)}")

    return model_file


def strip_model_quant(model_quant):
    model_stripped = deepcopy(model_quant)
    graph_conv.remove_float_inputs_outputs(model_stripped)
    graph_conv.remove_output_softmax(model_stripped)
    tflite2xcore_utils.clean_unused_opcodes(model_stripped)
    tflite2xcore_utils.clean_unused_tensors(model_stripped)
    tflite2xcore_utils.clean_unused_buffers(model_stripped)
    return model_stripped


def quantize_converter(converter, representative_data):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    x_train_ds = tf.data.Dataset.from_tensor_slices(representative_data).batch(1)

    def representative_data_gen():
        for input_value in x_train_ds.take(representative_data.shape[0]):
            yield [input_value]
    converter.representative_dataset = representative_data_gen


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


def save_test_data(data, data_dir, base_file_name):
    # save test data in numpy format
    test_data_dir = data_dir / base_file_name
    test_data_dir.mkdir(exist_ok=True, parents=True)
    np.savez(test_data_dir / f"{base_file_name}.npz", **data)

    # save individual binary files for easier low level access
    for key, test_set in data.items():
        for j, arr in enumerate(test_set):
            with open(test_data_dir / f"test_{j}.{key[0]}", 'wb') as f:
                f.write(arr.flatten().tostring())

    logging.info(f"test examples for {base_file_name} saved to {test_data_dir}")


def save_test_data_for_converter(converter, x_test_float, *,
                                 data_dir, base_file_name):
    # create interpreter
    interpreter = tf.lite.Interpreter(model_content=converter.convert())

    # extract reference labels for the test examples
    logging.info(f"Extracting examples for {base_file_name}...")
    y_test = apply_interpreter_to_examples(interpreter, x_test_float)
    data = {'x_test': x_test_float, 'y_test': np.vstack(y_test)}

    # save data
    save_test_data(data, data_dir, base_file_name)


def save_test_data_for_stripped_model(model_stripped, x_test_float, *,
                                      data_dir, base_file_name='model_stripped',
                                      add_float_outputs=True):
    model_stripped = deepcopy(model_stripped)

    # extract quantization info of input/output
    subgraph = model_stripped['subgraphs'][0]
    input_tensor = subgraph['tensors'][subgraph['inputs'][0]]
    input_quant = input_tensor['quantization']
    if add_float_outputs:
        output_tensor = subgraph['tensors'][subgraph['outputs'][0]]
        output_quant = output_tensor['quantization']

    # quantize test examples
    x_test = quantize(x_test_float, input_quant['scale'][0], input_quant['zero_point'][0])

    # add float interface
    graph_conv.add_float_inputs_outputs(model_stripped, outputs=add_float_outputs)

    # the TFLite interpreter needs a temporary file
    # the lifetime of this file needs to be at least the lifetime of the interpreter
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_tmp_file = save_from_json(
            model_stripped, pathlib.Path(tmp_dir), 'model_tmp', visualize=False)

        # create interpreter
        interpreter = tf.lite.Interpreter(model_path=str(model_tmp_file))

        # extract and quantize reference labels for the test examples
        logging.info(f"Extracting examples for {base_file_name}...")
        y_test = apply_interpreter_to_examples(interpreter, x_test_float)
        if add_float_outputs:
            y_test = map(
                lambda y: quantize(y, output_quant['scale'][0], output_quant['zero_point'][0]),
                y_test
            )
        data = {'x_test': x_test, 'y_test': np.vstack(list(y_test))}

    # save data
    save_test_data(data, data_dir, base_file_name)


def save_test_data_for_xcore_model(model_xcore, x_test_float, *,
                                   data_dir, base_file_name='model_xcore',
                                   pad_input_channel_dim=False):

    # extract quantization info of input/output
    subgraph = model_xcore['subgraphs'][0]
    input_tensor = subgraph['tensors'][subgraph['inputs'][0]]
    input_quant = input_tensor['quantization']

    if input_tensor['type'] == 'INT16':
        dtype = np.int16
    elif input_tensor['type'] == 'INT8':
        dtype = np.int8
    else:
        raise NotImplementedError(f"input tensor type {input_tensor['type']} "
                                  "not supported in save_test_data_for_xcore_model")

    # zero pad and quantize test examples
    if pad_input_channel_dim:
        old_shape = x_test_float.shape
        pad_shape = list(old_shape[:-1]) + [input_tensor['shape'][-1] - old_shape[-1]]
        pads = np.zeros(pad_shape, dtype=x_test_float.dtype)
        x_test_float = np.concatenate([x_test_float, pads], axis=3)
    x_test = quantize(x_test_float,
                      input_quant['scale'][0], input_quant['zero_point'][0], dtype)

    # save data
    save_test_data({'x_test': x_test}, data_dir, base_file_name)
