#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import re
import argparse

import xcore_model
import operators
from helpers import c_file, c_function

def indices2tensors(subgraph, tensors_indices):
    tensors = []

    for tensor_index in tensors_indices:
        tensors.append(subgraph.GetTensor(tensor_index))

    return tensors

def generate_code(args):
    verbose = args.verbose

    model = xcore_model.XCOREModel()
    model.Import(args.tflite_input, args.flatc, args.schema)
    subgraph = model.GetSubgraph() # only one supported for now

    nodes = subgraph.GetOperators()
    if verbose:
        print('*************')
        print('* Operators *')
        print('*************')
        for node in nodes:
            print(model.GetOperator(node.opcode_index))

    inputs = subgraph.GetInputs()
    if verbose:
        print('**********')
        print('* Inputs *')
        print('**********')
        for input_ in inputs:
            print(input_)

    initializers = [] # initializers are Tensors that are initialized with data
    variables = [] # variables are Tensors that are intermediates AND not initializers

    for intermediate in subgraph.GetIntermediates():
        buffer = model.GetBuffer(intermediate.GetBuffer())
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

    outputs = subgraph.GetOutputs()
    if verbose:
        print('***********')
        print('* Outputs *')
        print('***********')
        for output in outputs:
            print(output)

    # process the operators
    ops = []
    errs = []
    for node in nodes:
        try:
            name = model.GetOperator(node.GetOpcodeIndex())
            input_tensors = indices2tensors(subgraph, node.GetInputs())
            output_tensors = indices2tensors(subgraph, node.GetOutputs())
            op = operators.create(name, input_tensors, output_tensors)
            ops.append(op)
        except operators.UnsupportedOperator as err:
            errs.append(err)
        except operators.UnimplementedOperator as err:
            print(err)
    if errs:
        for err in errs:
            print(err, file=sys.stderr)
        sys.exit()

    # create function
    file_basename, _ = os.path.splitext(os.path.basename(args.tflite_input))
    fun_name = subgraph.GetName() or file_basename
    fun_name = re.sub('[^0-9a-zA-Z]+', '_', fun_name)

    fun = c_function.CFunction(fun_name,  inputs, outputs, variables)
    for op in ops:
        fun.add_operator(op)

    # create output
    includes = [
        'nn_operator.h'
    ]

    fd = c_file.CFile(args.name or fun_name, initializers=initializers,
        includes=includes, model=model)
    fd.add_function(fun)
    
    if args.name:
        fd.save()
    else:
        print(fd.render_source())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tflite_input', help='Input .tflite file.')
    parser.add_argument('-n', '--name', help='Output source files name')
    parser.add_argument('--flatc', required=False, default=None,
                        help='Path to flatc executable.')
    parser.add_argument('--schema', required=False, default=None,
                        help='Path to .fbs schema file.')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()

    generate_code(args)