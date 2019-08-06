#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import argparse

import onnx
from onnx import TensorProto, checker, optimizer

import operators
from helpers import c_file, c_function

#TODO: probably needs to be checked to make sure types mapping is correct
#        fix the complex types
#        add the quantized types (if ONNX supports those)
ONNX_TYPE_TO_C_TYPE = {
    int(TensorProto.FLOAT): 'float',
    int(TensorProto.INT8): 'int8',
    int(TensorProto.INT16): 'int16',
    int(TensorProto.INT32): 'int32',
    int(TensorProto.INT64): 'int64',
    int(TensorProto.UINT8): 'uint8',
    int(TensorProto.UINT16): 'uint16',
    # int(TensorProto.DOUBLE): 'double',
    # int(TensorProto.COMPLEX64): 'FIXME',
    # int(TensorProto.COMPLEX128): 'FIXME',
    # int(TensorProto.STRING): 'char',
    # int(TensorProto.BOOL): 'int8',
}

def optimize_model(model, verbose=False):
    if verbose:
        print('*********************************')
        print('* Available optimization passes *')
        print('*********************************')
        for p in optimizer.get_available_passes():
            print(p)


    # SEE: https://github.com/onnx/onnx/blob/master/onnx/optimizer.py for list of supported optimization pases
    passes = [
        'eliminate_nop_transpose',
        'eliminate_nop_pad',
        'fuse_consecutive_transposes',
        'fuse_consecutive_squeezes',
        'fuse_transpose_into_gemm',
        'fuse_add_bias_into_conv',
        'fuse_matmul_add_bias_into_gemm'
    ]

    optimized_model = optimizer.optimize(model, passes)

    # if verbose:
    #     print('*******************')
    #     print('* Optimized Model *')
    #     print('*******************')
    #     print(optimized_model)

    return optimized_model

def get_model_variables(model, verbose=False):
    variables = []

    # get input names
    inputs = set([])
    for value_info in model.graph.input:
        inputs.add(value_info.name)

    if verbose:
        print('*******************')
        print('* Model Variables *')
        print('*******************')

    for init in model.graph.initializer:
        # NOTE: When an initializer has the same name as a graph input, it specifies 
        #       a default value for that input. When an initializer has a name different 
        #       from all graph inputs, it specifies a constant value.
        variable = {
            'name': init.name,
            'type': ONNX_TYPE_TO_C_TYPE[init.data_type],
            'dims': init.dims,
            'values': [],  # set below
            'const': init.name not in inputs
        }

        # now set the values based on the data type
        if variable['type'] == 'float':
            variable['values'] = init.float_data
        elif variable['type'] == 'int32':
            variable['values'] = init.int32_data # ???
        elif variable['type'] == 'int16':
            variable['values'] = init.int16_data  # ???
        elif variable['type'] == 'int8':
            variable['values'] = init.int8_data  # ???

        variables.append(variable)
        if verbose:
            print(init.name)

    return variables

def get_model_intermediates(model, verbose=False):
    variables = []

    if verbose:
        print('***********************')
        print('* Model Intermediates *')
        print('***********************')

    for value_info in model.graph.value_info:
        dims = []
        for dim in value_info.type.tensor_type.shape.dim:
            dims.append(dim.dim_value)

        variable = {
            'name': value_info.name,
            'type': ONNX_TYPE_TO_C_TYPE[value_info.type.tensor_type.elem_type],
            'dims': dims
        }

        variables.append(variable)

        if verbose:
            print(value_info.name)

    return variables

def get_model_inputs(model, verbose=False):
    inputs = []

    if verbose:
        print('****************')
        print('* Model Inputs *')
        print('****************')

    # get initializer names
    initializers = set([])
    for init in model.graph.initializer:
        initializers.add(init.name)

    for value_info in model.graph.input:
        #print(value_info)
        if value_info.name not in initializers:
            c_type = ONNX_TYPE_TO_C_TYPE[value_info.type.tensor_type.elem_type]
            c_name = value_info.name
            #TODO: determine if value_input is array or scalar
            #        don't pass by reference if a scalar
            inputs.append({
                'name': c_name,
                'type': f'{c_type}',
                'scalar': False
            })
            
            if verbose:
                print(f'{c_name} {c_type}')

    return inputs

def get_model_outputs(model, verbose=False):
    outputs = []

    if verbose:
        print('*****************')
        print('* Model Outputs *')
        print('*****************')


    for value_info in model.graph.output:
        c_type = ONNX_TYPE_TO_C_TYPE[value_info.type.tensor_type.elem_type]
        c_name = value_info.name
        outputs.append({
            'name': c_name,
            'type': f'{c_type}'
        })

        if verbose:
            print(f'{c_name} {c_type}')

    return outputs

def get_operator_inputs(operator, verbose=False):
    inputs = []

    if verbose:
        print('*******************')
        print('* Operator Inputs *')
        print('*******************')

    for value_info in operator.input:
        print(value_info)
        if value_info.name not in initializers:
            c_type = ONNX_TYPE_TO_C_TYPE[value_info.type.tensor_type.elem_type]
            c_name = value_info.name
            #TODO: determine if value_input is array or scalar
            #        don't pass by reference if a scalar
            inputs.append({
                'name': c_name,
                'type': f'{c_type}',
                'scalar': False
            })
            
            if verbose:
                print(f'{c_name} {c_type}')

    return inputs

def get_operator_outputs(model, verbose=False):
    outputs = []

    if verbose:
        print('***********')
        print('* Outputs *')
        print('***********')


    for value_info in model.graph.output:
        c_type = ONNX_TYPE_TO_C_TYPE[value_info.type.tensor_type.elem_type]
        c_name = value_info.name
        outputs.append({
            'name': c_name,
            'type': f'{c_type}'
        })

        if verbose:
            print(f'{c_name} {c_type}')

    return outputs

def onnx2xc(args):

    # load model
    onnx_model = onnx.load(args.input)

    # verify the model
    checker.check_model(onnx_model)
    # TODO: probably want to verify opset_version and that all used operators are supported
    # SEE: https://github.com/onnx/onnx/blob/master/docs/IR.md

    # Apply the optimization on the original model
    onnx_model = optimize_model(onnx_model, args.verbose)

    # load variables
    variables = get_model_variables(onnx_model, args.verbose)
    # print the operators
    # op_types = set([])
    # for node in onnx_model.graph.node:
    #      op_types.add(node.op_type)
    #      print(node)
    #      print('-----------------------')
    # print()
    # print('op_types')
    # print(op_types)

    # process the operators
    ops = []
    errs = []
    for node in onnx_model.graph.node:
        try:
            op = operators.create(node.op_type, node.input, node.output, node.name)
            ops.append(op)
        except operators.UnsupportedOperator as err:
            errs.append(err)
        except operators.UnimplementedOperator as err:
            print(err)
    if errs:
        for err in errs:
            print(err, file=sys.stderr)
        sys.exit()

    # print()
    # print('input')
    # print(onnx_model.graph.input)
    # print()
    # print('output')
    # print(onnx_model.graph.output)

    # create function
    fun_name = onnx_model.graph.name
    fun_inputs = get_model_inputs(onnx_model, args.verbose)
    fun_outputs = get_model_outputs(onnx_model, args.verbose)
    fun_intermediates = get_model_intermediates(onnx_model, args.verbose)

    fun = c_function.CFunction(fun_name, fun_inputs, fun_outputs, fun_intermediates)
    for op in ops:
        fun.add_operator(op)

    # create output
    includes = [
        'lib_dsp.h',
        'lib_ai.h'
    ]
    fd = c_file.CFile(args.output or fun_name, variables=variables, includes=includes)
    fd.add_function(fun)
    
    if args.output:
        fd.save()
    else:
        print(fd.render_source())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input onnx file')
    parser.add_argument('-o', '--output', help='Output XC file')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()

    onnx2xc(args)