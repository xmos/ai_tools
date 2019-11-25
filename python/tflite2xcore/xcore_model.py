# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import struct

from operator_codes import OperatorCode

TFLITE_TYPE_TO_C_TYPE = {
    'FLOAT32': 'float32_t',
    'FLOAT16': 'float16_t',
    'INT32': 'int32_t',
    'UINT8': 'uint8_t',
    'INT64': 'int64_t',
    'INT16': 'int16_t',
    'UINT16': 'uint16_t',
    'INT8': 'int8_t',
    # 'STRING': 'TODO',
    # 'BOOL': 'TODO',
    # 'COMPLEX64': 'TODO?'
}

TFLITE_TYPE_TO_BYTES = {
    'FLOAT32': 4,
    'FLOAT16': 2,
    'INT32': 4,
    'UINT8': 1,
    'INT64': 8,
    'INT16': 2,
    'UINT16': 2,
    'INT8': 1,
    # 'STRING': 'TODO',
    # 'BOOL': 'TODO',
    # 'COMPLEX64': 'TODO?'
}


class Buffer():
    def __init__(self, model, data=None):
        # Generally, do not use this constructor to instantiate Buffer!
        # Use XCOREModel.create_buffer instead.

        self.model = model  # parent
        self.data = data or []

    def __str__(self):
        if self.data:
            len_ = len(self.data)
            return f'{len_}'
        else:
            return f'[]'

    def unpack(self, stdtype='uint8_t'):
        if stdtype == 'uint8_t':
            return [i[0] for i in struct.iter_unpack('B', bytearray(self.data))]
        elif stdtype == 'int8_t':
            return [i[0] for i in struct.iter_unpack('b', bytearray(self.data))]
        elif stdtype == 'int16_t':
            return [i[0] for i in struct.iter_unpack('h', bytearray(self.data))]
        elif stdtype == 'int32_t':
            return [i[0] for i in struct.iter_unpack('i', bytearray(self.data))]


class Operator():
    def __init__(self, subgraph, operator_code,
                 inputs=None, outputs=None, builtin_options=None, custom_options=None):
        # Generally, do not use this constructor to instantiate Operator!
        # Use Subgraph.create_operator instead.
        self.subgraph = subgraph  # parent
        self.operator_code = operator_code
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.builtin_options = builtin_options
        self.custom_options = custom_options

    @property
    def model(self):
        return self.subgraph.model

    def __str__(self):
        INDENT = ' ' * 2

        lines = []
        lines.append(f'operator_code= {self.operator_code}')
        lines.append(f'{INDENT}inputs')
        lines.extend([f'{INDENT * 2}{input_}' for input_ in self.inputs])
        lines.append(f'{INDENT}outputs')
        lines.extend([f'{INDENT * 2}{output}' for output in self.outputs])
        return '\n'.join(lines)


class Tensor():
    def __init__(self, subgraph, name, type_, shape, buffer=None, quantization=None):
        # Generally, do not use this constructor to instantiate Tensor!
        # Use Subgraph.create_tensor instead.
        self.subgraph = subgraph  # parent
        self.name = name
        self.type = type_
        self.shape = shape

        if buffer:
            isinstance(buffer, Buffer)
            assert buffer in self.model.buffers
            self.buffer = buffer
        else:
            self.buffer = self.model.create_buffer()

        self.quantization = quantization

    @property
    def model(self):
        return self.subgraph.model

    def __str__(self):
        return f'name={self.name}, type={self.type}, shape={self.shape}, buffer={self.buffer}'

    @property
    def sanitized_name(self):
        '''Return a name that is safe to use in source code'''
        return self.name.replace('/', '_')

    @property
    def name_segments(self):
        return self.name.split('/')

    @property
    def base_name(self):
        return self.name_segments()[-1]

    @property
    def standard_type(self):
        '''Return type (from cstdint.h)'''
        return TFLITE_TYPE_TO_C_TYPE[self.type]

    @property
    def size(self):
        size = TFLITE_TYPE_TO_BYTES[self.type]
        for s in self.shape:
            size *= s
        return size


class Subgraph():
    def __init__(self, model, inputs=None, outputs=None, operators=None, tensors=None):
        # Generally, do not use this constructor to instantiate Subgraph!
        # Use XCOREModel.create_subgraph instead.
        self.model = model  # parent
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.operators = operators or []
        self.tensors = tensors or []

    @property
    def intermediates(self):
        # intermediates are any tensors that are not an input or an output
        input_output_tensors = set([])  # lookup for input and output tensor indices
        for input_ in self.inputs:
            input_output_tensors.add(input_.name)
        for output in self.outputs:
            input_output_tensors.add(output.name)

        intermediates = []
        for tensor, tensor in enumerate(self.tensors):
            if tensor.name not in input_output_tensors:
                intermediates.append(tensor)

        return intermediates

    def create_tensor(self, name, type_, shape, *,
                      buffer=None, quantization=None, isinput=False, isoutput=False):
        for existing_tensor in self.tensors:
            if name in [existing_tensor.name, existing_tensor.sanitized_name]:
                raise Exception(f'Tensor name {name} already in use')

        tensor = Tensor(self, name, type_, shape, buffer, quantization)
        self.tensors.append(tensor)
        if isinput:
            self.inputs.append(tensor)
        if isoutput:
            self.outputs.append(tensor)
        return tensor

    def create_operator(self, operator_code, *,
                        inputs=None, outputs=None, builtin_options=None, custom_options=None):
        assert isinstance(operator_code, OperatorCode)
        operator = Operator(self, operator_code, inputs, outputs, builtin_options, custom_options)
        self.operators.append(operator)
        return operator

    def get_tensor(self, name):
        for t in self.tensors:
            if t.name == name:
                return t
        raise ValueError(f"Tensor with name {name} not found!")


class XCOREModel():
    def __init__(self, version=None, description=None, subgraphs=None, buffers=None, metadata=None):
        self.version = version
        self.description = description
        self.buffers = buffers or []
        self.subgraphs = subgraphs or []
        self.metadata = metadata

    def create_buffer(self, data=None):
        buffer = Buffer(self, data)
        self.buffers.append(buffer)
        return buffer

    def create_subgraph(self):
        subgraph = Subgraph(self)
        self.subgraphs.append(subgraph)
        return subgraph

    @property
    def operator_codes(self):
        operator_codes = set()

        for subgraph in self.subgraphs:
            for operator in subgraph.operators:
                operator_codes.add(operator.operator_code)

        return operator_codes

    def pprint(self):
        print('---------')
        print('- Model -')
        print('---------')
        print(f'description={self.description}')
        print(f'version={self.version}')
        print(f'metadata={self.metadata}')
        print('******************')
        print('* Operator Codes *')
        print('******************')
        for operator_code in self.operator_codes:
            print(operator_code)
        print('***********')
        print('* Buffers *')
        print('***********')
        for buffer in self.buffers:
            print(buffer)
        for subgraph in self.subgraphs:
            print('============')
            print('= Subgraph =')
            print('============')
            print('*************')
            print('* Operators *')
            print('*************')
            for operator in subgraph.operators:
                print(operator)

            print('**********')
            print('* Inputs *')
            print('**********')
            for input_ in subgraph.inputs:
                print(input_)

            print('*****************')
            print('* Intermediates *')
            print('*****************')
            for intermediate in subgraph.intermediates:
                print(intermediate)

            print('***********')
            print('* Outputs *')
            print('***********')
            for output in subgraph.outputs:
                print(output)
