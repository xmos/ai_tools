# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import struct
import enum
import collections

import numpy as np

from .operator_codes import OperatorCode


class TensorType(enum.Enum):
    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    UINT8 = 3
    INT64 = 4
    STRING = 5
    BOOL = 6
    INT16 = 7
    COMPLEX64 = 8
    INT8 = 9

    @staticmethod
    def to_stdint_type(tensor_type):
        LUT = {
            TensorType.FLOAT32: 'float32_t',
            TensorType.FLOAT16: 'float16_t',
            TensorType.INT32: 'int32_t',
            TensorType.UINT8: 'uint8_t',
            TensorType.INT64: 'int64_t',
            TensorType.STRING: None,
            TensorType.BOOL: 'uint8_t',
            TensorType.INT16: 'int16_t',
            TensorType.COMPLEX64: None,
            TensorType.INT8: 'int8_t'
        }
        return LUT[tensor_type]

    @staticmethod
    def to_bytes(tensor_type):
        LUT = {
            TensorType.FLOAT32: 4,
            TensorType.FLOAT16: 2,
            TensorType.INT32: 2,
            TensorType.UINT8: 1,
            TensorType.INT64: 8,
            TensorType.STRING: None,
            TensorType.BOOL: 1,
            TensorType.INT16: 2,
            TensorType.COMPLEX64: None,
            TensorType.INT8: 1
        }
        return LUT[tensor_type]

    @staticmethod
    def to_numpy_type(tensor_type):
        LUT = {
            TensorType.FLOAT32: np.float64,
            TensorType.FLOAT16: np.float64,
            TensorType.INT32: np.int64,
            TensorType.UINT8: np.int64,
            TensorType.INT64: np.int64,
            # TensorType.STRING: None,  # intentionally not supported
            TensorType.BOOL: np.int64,
            TensorType.INT16: np.int64,
            # TensorType.COMPLEX64: None,  # intentionally not supported
            TensorType.INT8: np.int64,
        }
        return LUT[tensor_type]


class Buffer():
    def __init__(self, model, data=None):
        # Generally, do not use this constructor to instantiate Buffer!
        # Use XCOREModel.create_buffer instead.

        self.model = model  # parent
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            self._data = []
        elif isinstance(data, list):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = list(data.flatten().tostring())
        else:
            raise TypeError(f"data must be list or numpy array")

    def __len__(self):
        if self.data:
            return len(self.data)
        else:
            return 0

    def __str__(self):
        if self.data:
            len_ = len(self.data)
            return f'{len_}'
        else:
            return f'[]'

    def unpack(self, stdtype='uint8_t'):
        LUT = {'uint8_t': 'B',
               'int8_t': 'b',
               'int16_t': 'h',
               'int32_t': 'i'}
        return [i[0] for i in struct.iter_unpack(LUT[stdtype], bytearray(self.data))]


class Operator():
    def __init__(self, subgraph, operator_code,
                 name=None, inputs=None, outputs=None,
                 builtin_options=None, builtin_options_type=None, custom_options=None):
        # Generally, do not use this constructor to instantiate Operator!
        # Use Subgraph.create_operator instead.
        assert isinstance(operator_code, OperatorCode)

        self.subgraph = subgraph  # parent
        self.operator_code = operator_code
        self.name = name
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.builtin_options = builtin_options
        self.builtin_options_type = builtin_options_type
        if builtin_options:
            assert builtin_options_type
        self.custom_options = custom_options

    def add_custom_options(self, **kwargs):
        if kwargs:
            if self.custom_options is None:
                self.custom_options = {}
            self.custom_options.update(kwargs)

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
        lines.append(f'{INDENT}builtin_options')
        lines.append(f'{INDENT * 2}{self.builtin_options}')
        lines.append(f'{INDENT}custom_options')
        lines.append(f'{INDENT * 2}{self.custom_options}')
        return '\n'.join(lines)


class Tensor():
    def __init__(self, subgraph, name, type_, shape, buffer=None, quantization=None):
        # Generally, do not use this constructor to instantiate Tensor!
        # Use Subgraph.create_tensor instead.
        self.subgraph = subgraph  # parent
        self.name = name
        assert isinstance(type_, TensorType)
        self.type = type_
        self.shape = list(shape)

        if buffer:
            assert isinstance(buffer, Buffer)
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
        return TensorType.to_stdint_type(self.type)

    @property
    def size(self):
        size = TensorType.to_bytes(self.type)
        for s in self.shape:
            size *= s
        return size

    @property
    def numpy(self):
        arr = np.array(
            self.buffer.unpack(TensorType.to_stdint_type(self.type)),
            dtype=TensorType.to_numpy_type(self.type)
        )
        return arr.reshape(self.shape)


class Subgraph():
    def __init__(self, model, name=None, inputs=None, outputs=None, operators=None, tensors=None):
        # Generally, do not use this constructor to instantiate Subgraph!
        # Use XCOREModel.create_subgraph instead.
        self.model = model  # parent
        self.name = name
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.operators = operators or []
        self.tensors = tensors or []

    @property
    def intermediates(self):
        # intermediates are any tensors that are not an input or an output
        return [t for t in self.tensors if t not in (self.inputs + self.outputs)]

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

    def remove_tensor(self, tensor):
        assert tensor in self.tensors
        self.tensors.remove(tensor)
        if tensor in self.inputs:
            self.inputs.remove(tensor)
        if tensor in self.outputs:
            self.outputs.remove(tensor)

        tensor.subgraph = tensor.buffer = None

    def generate_unique_op_name(self, operator_code):
        existing_names = [op.name for op in self.operators]
        j = 0
        while True:
            j, new_name = j+1, f"{operator_code.name}_{j}"
            if new_name not in existing_names:
                return new_name

    def create_operator(self, operator_code, *,
                        inputs=None, outputs=None,
                        builtin_options=None, builtin_options_type=None, custom_options=None):
        name = self.generate_unique_op_name(operator_code)
        operator = Operator(self, operator_code, name, inputs, outputs,
                            builtin_options, builtin_options_type, custom_options)
        self.operators.append(operator)
        return operator

    def remove_operator(self, op):
        assert op in self.operators
        self.operators.remove(op)
        op.inputs, op.outputs, op.subgraph = [], [], None

    def get_tensor(self, name):
        for t in self.tensors:
            if t.name == name:
                return t
        raise ValueError(f"Tensor with name {name} not found!")


class Metadata():
    def __init__(self, model, name, buffer=None):
        # Generally, do not use this constructor to instantiate Metadata!
        # Use XCOREModel.create_metadata instead.
        self.model = model  # parent
        self.name = name
        if buffer:
            assert isinstance(buffer, Buffer)
            assert buffer in self.model.buffers
            self.buffer = buffer
        else:
            self.buffer = self.model.create_buffer()

    def __str__(self):
        return f'name={self.name}, buffer={self.buffer}'


class XCOREModel():
    def __init__(self, version=None, description=None, subgraphs=None, buffers=None, metadata=None):
        self.version = version or 3
        self.description = description or ''
        self.buffers = buffers or []
        self.subgraphs = subgraphs or []
        self.metadata = metadata or []

    def create_buffer(self, data=None):
        buffer = Buffer(self, data)
        self.buffers.append(buffer)
        return buffer

    def create_metadata(self, name, buffer=None):
        metadata = Metadata(self, name, buffer)
        self.metadata.append(metadata)
        return metadata

    def create_subgraph(self, name=None):
        subgraph = Subgraph(self, name)
        self.subgraphs.append(subgraph)
        return subgraph

    @property
    def operator_codes(self):
        # sort the operators codes from most frequent to least frequent
        #   why? because the flatbuffer is a tiny bit smaller if we do
        counter = collections.Counter()

        for subgraph in self.subgraphs:
            for operator in subgraph.operators:
                counter[operator.operator_code] += 1

        sorted_operator_codes = [op_code for op_code, _ in counter.most_common()]

        return sorted_operator_codes

    def pprint(self, tensor_values=False):
        print('---------')
        print('- Model -')
        print('---------')
        print(f'description={self.description}')
        print(f'version={self.version}')
        print('******************')
        print('* Metadata *')
        print('******************')
        for metadata in self.metadata:
            print(metadata)
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
                if tensor_values and len(input_.buffer):
                    print(f'   values={input_.numpy}')

            print('*****************')
            print('* Intermediates *')
            print('*****************')
            for intermediate in subgraph.intermediates:
                print(intermediate)
                if tensor_values and len(intermediate.buffer):
                    print(f'   values={intermediate.numpy}')

            print('***********')
            print('* Outputs *')
            print('***********')
            for output in subgraph.outputs:
                print(output)
                if tensor_values and len(output.buffer):
                    print(f'   values={output.numpy}')
