#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import sys
import logging
import argparse

import tflite2xcore.converter as xcore_conv
from tflite2xcore.model_generation import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_input', help='Input .tflite file.')
    parser.add_argument('tflite_output', help='Output .tflite file.')
    parser.add_argument('--classifier', action='store_true', default=False,
                        help="Apply optimizations for classifier networks "
                             "(e.g. softmax removal and output argmax).")
    parser.add_argument('--remove_softmax', action='store_true', default=False,
                        help="Remove output softmax operation.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    utils.set_gpu_usage(False, verbose)

    tflite_input_path = os.path.realpath(args.tflite_input)
    tflite_output_path = os.path.realpath(args.tflite_output)
    is_classifier = args.classifier
    remove_softmax = args.remove_softmax

    xcore_conv.convert(tflite_input_path, tflite_output_path,
         is_classifier=is_classifier, remove_softmax=remove_softmax)
