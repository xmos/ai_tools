#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import subprocess


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

fn = 'test.bin'
T = 0

# make test tensors and output
for i, truth_label in enumerate(y_test):
    orig_img = x_test[i]
    # make signed int8
    signed_img = np.ndarray.astype(orig_img, 'int8')
    # pad
    padded_img = np.pad(signed_img, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
    # flatten
    flattened_img = padded_img.flatten()
    # output
    with open(fn, 'wb') as fd:
        fd.write(flattened_img.tobytes())

    cmd = 'xsim --args ../codegen/bin/800MHz/codegen_800MHz.xe test.bin'
    #cmd = 'xsim --args ../tflite/bin/800MHz/tflite_800MHz.xe test.bin'
    xsim_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    lines = xsim_output.decode('utf-8').split()
    fields = lines[-1].split('=')

    #print(xsim_output)

    predicted_label = fields[-1].strip()
    truth_label = labels[truth_label[0]]
    if predicted_label == truth_label:
        T = T + 1
    N = i + 1
    P = T/N * 100
    print(f'{i}: truth={truth_label}   predicted={predicted_label}   accuracy={T}/{N}  {P:.2f}%')

os.remove(fn)
