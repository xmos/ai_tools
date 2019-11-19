#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import argparse

import xcore_model

def print_subgraph(model, subgraph):
    nodes = subgraph.operators
    print('*************')
    print('* Operators *')
    print('*************')
    for node in nodes:
        print(model.get_operator_code(node.opcode_index))

    inputs = subgraph.inputs
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

    print('****************')
    print('* Initializers *')
    print('****************')
    for initializer in initializers:
        print(initializer)

    print('*************')
    print('* Variables *')
    print('*************')
    for variable in variables:
        print(variable)

    outputs = subgraph.outputs
    print('***********')
    print('* Outputs *')
    print('***********')
    for output in outputs:
        print(output)

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
    print_subgraph(model, subgraph)

    # Now do some editing
    print('')
    print('')
    print('---------------------------------')
    print('- Trim first and last operators -')
    print('---------------------------------')
    print('')
    print('')
    first_operator = subgraph.operators[0]
    first_operator.trim()  # remove on the operator object
    last_operator = subgraph.operators[-1]
    subgraph.trim_operator(last_operator) # or, remove on the subgraph opject
    print_subgraph(model, subgraph)
    print('')
    print('')
    print('---------------------------------')
    print('- Substitute an operator        -')
    print('---------------------------------')
    print('')
    print('')
    new_operator_code = model.create_operator_code(custom_code='XC_CONV2D')
    
    new_operator = subgraph.create_operator(new_operator_code)
    subgraph.substitute_operator(new_operator_code)

    new_buffer = model.create_buffer([1] * 1 * 5 * 5 * 3)
    new_tensor = subgraph.create_tensor(
        'test/new_tensor',
        'INT8',
        [1, 5, 5, 3],
        new_buffer
    )
    print_subgraph(model, subgraph)
    print()
    print(new_operator_code)
    print(new_tensor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_input', help='Input .tflite file.')
    parser.add_argument('--flatc', required=False, default=None,
                        help='Path to flatc executable.')
    parser.add_argument('--schema', required=False, default=None,
                        help='Path to .fbs schema file.')
    args = parser.parse_args()

    test_xcore_model(args)