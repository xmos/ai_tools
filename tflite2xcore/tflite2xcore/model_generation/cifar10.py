# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

import numpy as np
from .utils import tf


def get_normalized_data():
    (
        (train_images, train_labels),
        (test_images, test_labels),
    ) = tf.keras.datasets.cifar10.load_data()

    scale = tf.constant(255, dtype=tf.dtypes.float32)
    x_train, x_test = train_images / scale - 0.5, test_images / scale - 0.5
    y_train, y_test = train_labels, test_labels

    return {
        "x_train": np.float32(x_train),
        "y_train": np.float32(y_train),
        "x_test": np.float32(x_test),
        "y_test": np.float32(y_test),
    }
