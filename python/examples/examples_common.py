# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import random
import logging

import numpy as np

from copy import deepcopy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
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


def quantize(arr, scale, zero_point):
    t = np.round(arr / scale + zero_point)
    return np.int8(np.round(np.clip(t, -128, 127)))


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
