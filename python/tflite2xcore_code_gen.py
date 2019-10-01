#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import argparse

import xcore_model

def generate_code(args):
    verbose = args.verbose

    model = xcore_model.XCOREModel()
    model.Import(args.tflite_input, args.flatc, args.schema)
    subgraph = model.GetSubgraph()

    operators = subgraph.GetOperators()
    if verbose:
        print('*************')
        print('* Operators *')
        print('*************')
        for operator in operators:
            print(model.GetOperator(operator.opcode_index))

    inputs = subgraph.GetInputs()
    if verbose:
        print('**********')
        print('* Inputs *')
        print('**********')
        for input_ in inputs:
            print(input_)

    intermediates = subgraph.GetIntermediates()
    if verbose:
        print('*****************')
        print('* Intermediates *')
        print('*****************')
        for intermediate in intermediates:
            print(intermediate)

    outputs = subgraph.GetOutputs()
    if verbose:
        print('***********')
        print('* Outputs *')
        print('***********')
        for output in outputs:
            print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_input', help='Input .tflite file.')
    parser.add_argument('-n', '--name', required=True, help='Output source files name')
    parser.add_argument('--flatc', required=False, default=None,
                        help='Path to flatc executable.')
    parser.add_argument('--schema', required=False, default=None,
                        help='Path to .fbs schema file.')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()

    generate_code(args)