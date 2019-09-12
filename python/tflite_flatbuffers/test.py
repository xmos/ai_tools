#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys

import flatbuffers
from tflite.Model import *
from tflite.BuiltinOperator import BuiltinOperator
from tflite.TensorType import TensorType


def make_operator_lookup():
# makes an array to lookup operator name given an enum
    bio = BuiltinOperator()
    ops = [a for a in dir(bio) if not a.startswith('__')]

    lookup = [-1] * len(ops)
    for op in ops:
        lookup[getattr(bio, op)] = op

    return lookup

def make_type_lookup():
# makes an array to lookup data type name given an enum
    tt = TensorType()
    types = [a for a in dir(tt) if not a.startswith('__')]

    lookup = [-1] * len(types)
    for ty in types:
        lookup[getattr(tt, ty)] = ty

    return lookup

# NOTE: This fuction does not currently do a full import of all opjects in the model
#       For reference, the C++ Import method is here:
#       https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/toco/tflite/import.cc
def import_model(model_filename):
    buf = open(model_filename, "rb").read()
    buf = bytearray(buf)
    model = Model.GetRootAsModel(buf, 0)

    operator_lookup = make_operator_lookup()
    type_lookup = make_type_lookup()

    used_op_codes = []

    print('*********')
    print('* Model *')
    print('*********')
    print(f'Model attributes: {dir(model)}')
    print(f'Version: {model.Version()}')
    print(f'Description: {model.Description().decode("utf-8")}')
    for i in range(model.OperatorCodesLength()):
        op_code = model.OperatorCodes(i)
        used_op_codes.append(op_code.BuiltinCode()) # assuming all operators are builtin
        print(f'OperatorCodes: Builtin={op_code.BuiltinCode()}, Custom={op_code.CustomCode()}')
    print('***********')
    print('* Buffers *')
    print('***********')
    for index, i in enumerate(range(model.BuffersLength())):
        print('**********')
        print('* Buffer *')
        print('**********')
        buffer = model.Buffers(i)
        print(f'Index: {index}')
        print(f'Length (bytes): {buffer.DataLength()}')
        #print(f'Data: {buffer.DataAsNumpy()}')
    print('*************')
    print('* Subgraphs *')
    print('*************')
    for i in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(i)
        print(f'Subgraph Attributes: {dir(subgraph)}')
        print('********************')
        print('* Subgraph Tensors *')
        print('********************')
        tensors = []
        for index, j in enumerate(range(subgraph.TensorsLength())):
            print('**********')
            print('* Tensor *')
            print('**********')
            tensor = subgraph.Tensors(j)
            print(f'Tensor Attributes: {dir(tensor)}')
            print(f'Index: {index}')
            print(f'Name: {tensor.Name().decode("utf-8")}')
            tensor_type = type_lookup[tensor.Type()]
            print(f'Type: {tensor_type}')
            shape = tensor.ShapeAsNumpy()
            print(f'Shape: {shape}')
            tensors.append(tensor)
            print(f'Buffer Index: {tensor.Buffer()}')
        print('*******************')
        print('* Subgraph Inputs *')
        print('*******************')
        input_tensors = subgraph.InputsAsNumpy()
        print(f'Input Tensor Indices: {input_tensors}')
        print('********************')
        print('* Subgraph Outputs *')
        print('********************')
        output_tensors = subgraph.OutputsAsNumpy()
        print(f'Ouputs Tensor Indices: {output_tensors}')
        print('**********************')
        print('* Subgraph Operators *')
        print('**********************')
        for j in range(subgraph.OperatorsLength()):
            print('************')
            print('* Operator *')
            print('************')
            operator = subgraph.Operators(j)
            print(f'Operator Attributes: {dir(operator)}')
            print(f'OpcodeIndex: {operator.OpcodeIndex()}')
            op_code = used_op_codes[operator.OpcodeIndex()]
            print(f'Operator: {operator_lookup[op_code]}')
            #TODO: print out other bits on operator, including operator options?
            print('*******************')
            print('* Operator Inputs *')
            print('*******************')
            input_tensors = operator.InputsAsNumpy()
            print(f'Input Tensor Indices: {input_tensors}')
            # print('**************************')
            # print('* Operator Intermediates *')
            # print('**************************')
            # intermediate_tensors = operator.IntermediatesAsNumpy()
            # print(f'Intermediate Tensor Indices: {intermediate_tensors}')
            print('********************')
            print('* Operator Outputs *')
            print('********************')
            output_tensors = operator.OutputsAsNumpy()
            print(f'Ouputs Tensor Indices: {output_tensors}')

# NOTE: This export function is a placeholder for an actual export to be implemented later
#       For reference, the C++ Export method is here:
#       https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/toco/tflite/export.cc
def export_model(model_filename):
    builder = flatbuffers.Builder(0)

    ModelStart(builder)
    #TODO: serialize full model

    ModelEnd(builder)

    buf = builder.Bytes
    with open(model_filename, 'wb') as fd:
        fd.write(builder.Bytes)

if __name__ == "__main__":

    model_filename = sys.argv[1]

    model = import_model(model_filename)

    # test saving
    # export_model('test.tflite')


