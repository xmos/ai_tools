#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import argparse

import xcore_model

def test_xcore_model(args):
    verbose = args.verbose

    model = xcore_model.XCOREModel()
    model.Import(args.tflite_input, args.flatc, args.schema)
    subgraph = model.get_subgraph() # only one supported for now

    nodes = subgraph.operators
    if verbose:
        print('*************')
        print('* Operators *')
        print('*************')
        for node in nodes:
            print(model.get_operator(node.opcode_index))

    inputs = subgraph.inputs
    if verbose:
        print('**********')
        print('* Inputs *')
        print('**********')
        for input_ in inputs:
            print(input_)

    initializers = [] # initializers are Tensors that are initialized with data
    variables = [] # variables are Tensors that are intermediates AND not initializers

    for intermediate in subgraph.intermediates:
        buffer = model.get_buffer(intermediate.buffer)
        if buffer:
            initializers.append(intermediate)
        else:
            variables.append(intermediate)

    if verbose:
        print('****************')
        print('* Initializers *')
        print('****************')
        for initializer in initializers:
            print(initializer)

    if verbose:
        print('*************')
        print('* Variables *')
        print('*************')
        for variable in variables:
            print(variable)

    outputs = subgraph.outputs
    if verbose:
        print('***********')
        print('* Outputs *')
        print('***********')
        for output in outputs:
            print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_input', help='Input .tflite file.')
    parser.add_argument('--flatc', required=False, default=None,
                        help='Path to flatc executable.')
    parser.add_argument('--schema', required=False, default=None,
                        help='Path to .fbs schema file.')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()

    test_xcore_model(args)