#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import sys
import subprocess
import multiprocessing
import argparse

import tensorflow as tf
import numpy as np

LABELS = [
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
NUM_LABELS = len(LABELS)
XE_FILE = None

def run_testcase(testcase):
    global XE_FILE

    tmp_img = np.ndarray.astype(testcase, 'int16')
    # make signed int8
    signed_img = np.ndarray.astype((tmp_img-128), 'int8')
    # pad
    padded_img = np.pad(signed_img, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
    # flatten
    flattened_img = padded_img.flatten()
    # output
    cp = multiprocessing.current_process()
    with open(cp.name, 'wb') as fd:
        fd.write(flattened_img.tobytes())
    # run xsim & process output
    cmd = f'xsim --args {XE_FILE} {cp.name}'
    xsim_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    lines = xsim_output.decode('utf-8').split()
    fields = lines[-1].split('=')

    predicted_label = fields[-1].strip()

    os.remove(cp.name)

    return predicted_label

def chunks(l, n):
    for i in range(0, len(l), n):
        yield ([i for i in range(i, i + n)], l[i:i + n])

def test_accuracy(args):
    global XE_FILE
    XE_FILE = args.xe

    # load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    T = 0
    N = 0

    process_pool = multiprocessing.Pool(args.pool)
    for truth_indices, x_test_chunk in chunks(x_test, args.pool):
        predicted_labels = process_pool.map(run_testcase, x_test_chunk)
        for predicted_index, truth_index in enumerate(truth_indices):
            truth_label = LABELS[y_test[truth_index][0]]
            predicted_label = predicted_labels[predicted_index]
            if predicted_label == truth_label:
                T = T + 1
            N = N + 1
            P = T/N * 100
            print(f'truth={truth_label}   predicted={predicted_label}   accuracy={T}/{N}  {P:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xe', required=True, help='Input .xe file')
    parser.add_argument('-p', '--pool', type=int, default=4, help='Pool size')
    args = parser.parse_args()

    test_accuracy(args)
