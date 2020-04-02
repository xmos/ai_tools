#!/usr/bin/env python
#
# Copyright (c) 2019, XMOS Ltd, All rights reserved

import logging
import argparse

from pathlib import Path

from tflite2xcore import utils
import tflite2xcore.converter as xcore_conv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_input', help='Input .tflite file.')
    parser.add_argument('tflite_output', help='Output .tflite file.')
    parser.add_argument('--classifier', action='store_true', default=False,
                        help="Apply optimizations for classifier networks "
                             "(e.g. softmax removal and output argmax).")
    parser.add_argument('--remove_softmax', action='store_true', default=False,
                        help="Remove output softmax operation.")
    parser.add_argument('--minify', action='store_true', default=False,
                        help="Make the model smaller at the expense of readability.")
    parser.add_argument(
        '-par', '--num_threads', type=int, default=1,
        help='Number of parallel threads for xcore.ai optimization.'
    )
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose mode.')
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    tflite_input_path = Path(args.tflite_input)
    tflite_output_path = Path(args.tflite_output)

    xcore_conv.convert(tflite_input_path, tflite_output_path,
                       is_classifier=args.classifier,
                       remove_softmax=args.remove_softmax,
                       minification=args.minify,
                       num_threads=args.num_threads)
