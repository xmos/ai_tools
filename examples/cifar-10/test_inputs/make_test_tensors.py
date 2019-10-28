#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import tensorflow as tf
import numpy as np

labels = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck',
]
n_lables = len(labels)

# load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# find first example of each label in dataset
test_indices = [-1] * n_lables
for i_label in range(n_lables):
        where = np.where(y_test == i_label)
        test_indices[i_label] = where[0][0]

# make test tensors and output
for i_label, i_test in enumerate(test_indices):
    orig_img = x_test[i_test]
    # output orig
    # print(orig_img.dtype)
    # fn = '{}.orig'.format(labels[i_label])
    # with open(fn, 'wb') as fd:
    #     fd.write(orig_img.flatten().tobytes())
    # make signed int8
    signed_img = np.ndarray.astype(orig_img, 'int8')
    # pad
    padded_img = np.pad(signed_img, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
    # flatten
    flattened_img = padded_img.flatten()
    # output quantized
    fn = '{}.bin'.format(labels[i_label])
    with open(fn, 'wb') as fd:
        fd.write(flattened_img.tobytes())
