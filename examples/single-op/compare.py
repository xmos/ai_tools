#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import argparse

import numpy as np

def compare(args):
    expected = np.fromfile(args.expected_output, dtype='int8') 

    computed = np.fromfile(args.computed_output, dtype='int8')
    len_ratio = len(computed) / len(expected)
    if len_ratio == 2:
        computed = np.fromfile(args.computed_output, dtype='int16')
    elif len_ratio == 4:
        computed = np.fromfile(args.computed_output, dtype='int32')

    dequantized_expected = (expected - args.expected_zero_point) * args.expected_scale
    dequantized_computed = (computed - args.computed_zero_point) * args.computed_scale

    print('expected, computed')
    for e, c in zip(dequantized_expected, dequantized_computed):
        print(e, c)

    # computed = np.transpose(computed.reshape((5,5,16)), axes=(2,0,1))
    # computed = computed.reshape((1,5,16))
    # print(computed)
    # print('------------------------------------------')
    # expected = expected.reshape((1,5,16))
    # print(expected)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('expected_output', help='Expected output file.')
    parser.add_argument('-a', '--expected_zero_point', required=False, type=int, default=0,
                        help='Expected output zero point.')
    parser.add_argument('-b', '--expected_scale', required=False, type=float, default=1.0,
                        help='Expected output scale.')
    parser.add_argument('computed_output', help='Computed output file.')
    parser.add_argument('-c', '--computed_zero_point', required=False, type=int, default=0,
                        help='Computed zero point.')
    parser.add_argument('-d', '--computed_scale', required=False, type=float, default=1.0,
                        help='Computed scale.')
    args = parser.parse_args()

    compare(args)