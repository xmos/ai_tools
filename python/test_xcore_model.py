#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import argparse

import xcore_model

def test_xcore_model(args):
    model = xcore_model.XCOREModel()
    model.load(args.tflite_input, args.flatc, args.schema)
    subgraph = model.subgraphs[0]  # only one supported for now

    print('')
    print('')
    print('----------------------------------')
    print('- Original model                 -')
    print('----------------------------------')
    print('')
    print('')
    model.pprint()

    # Now do some editing
    print('')
    print('')
    print('-----------------------------------')
    print('- Remove first and last operators -')
    print('-----------------------------------')
    print('')
    print('')
    first_operator = subgraph.operators[0]
    subgraph.operators.remove(first_operator)
    last_operator = subgraph.operators[-1]
    subgraph.operators.remove(last_operator)
    model.pprint()
    print('')
    print('')
    print('---------------------------------')
    print('- Add an operator               -')
    print('---------------------------------')
    print('')
    print('')
    buffer1 = model.create_buffer([1]*123)
    model.buffers.append(buffer1)
    tensor1 = xcore_model.Tensor(
        model,
        subgraph,
        'test/new_tensor1',
        'INT8',
        [1, 1, 1, 123],
        buffer1
    )
    subgraph.tensors.append(tensor1)

    buffer2 = model.create_buffer()
    tensor2 = subgraph.create_tensor(
        'test/new_tensor2',
        'INT8',
        [1, 5, 5, 3],
        buffer2
    )
    operator1_code = {'builtin_code': 'CUSTOM', 'custom_code': 'FIZZBUZZ_OPERATOR', 'version': 1}
    operator1 = subgraph.create_operator(operator1_code)
    operator1.inputs.append(tensor1)
    operator1.inputs.append(subgraph.outputs[0])
    operator1.outputs.append(tensor2)
    subgraph.operators.append(operator1)
    # fixup subgraph output
    subgraph.outputs[0] = tensor2
    model.pprint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_input', help='Input .tflite file.')
    parser.add_argument('--flatc', required=False, default=None,
                        help='Path to flatc executable.')
    parser.add_argument('--schema', required=False, default=None,
                        help='Path to .fbs schema file.')
    args = parser.parse_args()

    test_xcore_model(args)