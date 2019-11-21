#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import re
import argparse

import tflite2xcore
import operators
from helpers import c_file, c_function

def generate_code(args):
    verbose = args.verbose

    model = tflite2xcore.read_flatbuffers_json(args.tflite_input, args.flatc, args.schema)

    subgraph = model.subgraphs[0] # only one supported for now
    intermediates_memory = 0
    total_memory = 0

    if verbose:
        model.pprint()

    for input_ in subgraph.inputs:
        total_memory += input_.size

    initializers = [] # initializers are Tensors that are initialized with data
    variables = [] # variables are Tensors that are intermediates AND not initializers

    for intermediate in subgraph.intermediates:
        data = intermediate.buffer.data
        if data:
            initializers.append(intermediate)
        else:
            variables.append(intermediate)
            intermediates_memory += intermediate.size
        total_memory += intermediate.size

    # process the operators
    ops = []
    errs = []
    for operator in subgraph.operators:
        try:
            name = operator.operator_code
            op = operators.create(name, operator.inputs, operator.outputs, model)
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
    #file_basename, _ = os.path.splitext(os.path.basename(args.tflite_input))
    fun_name = subgraph.name or args.name
    fun_name = re.sub('[^0-9a-zA-Z]+', '_', fun_name)

    fun = c_function.CFunction(fun_name, subgraph.inputs, subgraph.outputs)
    for op in ops:
        fun.add_operator(op)

    # create output
    includes = [
        'nn_operator.h'
    ]

    fd = c_file.CFile(args.name or fun_name, initializers=initializers,
        variables=variables, includes=includes, model=model)
    fd.add_function(fun)
    
    if args.name:
        fd.save()
    else:
        print(fd.render_source())

    # output some summary info
    print('Created files:')
    for fn in fd.get_filenames():
        print(f'   {fn}')

    intermediates_memory /= 1000.0
    print(f'Total memory used for intermediate tensors: {intermediates_memory} kB')

    total_memory /= 1000.0
    print(f'Total memory used for model: {total_memory} kB')

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