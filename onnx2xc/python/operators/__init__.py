# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

from operators import conv
from operators import reshape
from operators import squeeze
from operators import add
from operators import relu
from operators import maxpool
from operators import matmul
from operators import gemm

class UnsupportedOperator(Exception):
    def __init__(self, name):
        super().__init__(f'Operator {name} not supported.')

class UnimplementedOperator(Exception):
    def __init__(self, name):
        super().__init__(f'Operator {name} not implemented.')

#SEE: https://github.com/onnx/onnx/blob/master/docs/Operators.md

def create(name, inputs=None, outputs=None, commment=None):
    if name == 'Conv':
        return conv.Conv(inputs, outputs, commment)
    elif name == 'Reshape':
        return reshape.Reshape()
    elif name == 'Relu':
        return relu.Relu()
    elif name == 'Add':
        return add.Add()
    elif name == 'MatMul':
        return matmul.MatMul()
    elif name == 'MaxPool':
        return maxpool.MaxPool()
    elif name == 'Gemm':
        return gemm.Gemm()
    elif name == 'Squeeze':
        return squeeze.Squeeze()
    else:
        raise UnsupportedOperator(name)
