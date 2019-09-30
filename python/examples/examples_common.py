# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
warnings.filterwarnings(action='default')

import numpy as np


DEFAULT_SEED = 123


def set_all_seeds(seed=DEFAULT_SEED):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_aux_dirs(dirname):
    models_dir = dirname / "models"
    data_dir = dirname / "data"
    models_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)

    return models_dir, data_dir


def load_scaled_cifar10():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    scale = tf.constant(255, dtype=tf.dtypes.float32)
    x_train, x_test = train_images/scale - .5, test_images/scale - .5
    y_train, y_test = train_labels, test_labels

    return (x_train, y_train), (x_test, y_test)


def quantize_data(arr, scale, zero_point):
    t = np.round(arr / scale + zero_point)
    return np.int8(np.round(np.clip(t, -128, 127)))


def one_hot_encode(arr, classes):
    return tf.keras.utils.to_categorical(arr, classes, dtype=np.int8)