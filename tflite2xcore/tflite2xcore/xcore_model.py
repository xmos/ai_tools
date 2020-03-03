# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import sys
import struct
import enum
import collections
import logging

import numpy as np

from tflite2xcore.operator_codes import OperatorCode


class TensorType(enum.IntEnum):
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
    def __init__(self, model, data=None, *, owners=None):
        # Generally, do not use this constructor to instantiate Buffer!
        # Use XCOREModel.create_buffer instead.

        self.model = model  # parent
        self.data = data
        self.owners = owners or []

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data is None:
            self._data = np.array([], dtype=np.uint8)
        elif isinstance(data, list):
            self._data = np.array(data, dtype=np.uint8)
        elif isinstance(data, np.ndarray):
            if data.dtype not in (np.uint8, 'uint8'):
                logging.debug(f"Numpy array of type {data.dtype} stored in buffer")
            self._data = np.frombuffer(data.tostring(), dtype=np.uint8)
        else:
            raise TypeError(f"data must be list or numpy array of uint8 type")

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        else:
            return 0

    def __str__(self):
        if self.data is not None:
            return f'Buffer[{len(self.data)}]'
        else:
            return 'Buffer[]'

    def unpack(self, stdtype='uint8_t'):
        LUT = {'uint8_t': 'B',
               'int8_t': 'b',
               'int16_t': 'h',
               'int32_t': 'i',
               'float32_t': 'f'}
        return [i[0] for i in struct.iter_unpack(LUT[stdtype], bytearray(self.data))]

    def sanity_check(self):
        assert self in self.model.buffers
        for owner in self.owners:
            assert owner.buffer is self


class Operator():
    def __init__(self, subgraph, operator_code,
                 name=None, inputs=None, outputs=None,
                 builtin_options=None, builtin_options_type=None,
                 custom_options=None):
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
        return f'({self.subgraph.operators.index(self)}) operator_code={self.operator_code}'

    def pprint(self):
        INDENT = ' ' * 2

        lines = []
        lines.append(str(self))
        lines.append(f'{INDENT}inputs')
        lines.extend([f'{INDENT * 2}{input_}' for input_ in self.inputs])
        lines.append(f'{INDENT}outputs')
        lines.extend([f'{INDENT * 2}{output}' for output in self.outputs])
        lines.append(f'{INDENT}builtin_options')
        lines.append(f'{INDENT * 2}{self.builtin_options}')
        lines.append(f'{INDENT}custom_options')
        lines.append(f'{INDENT * 2}{self.custom_options}')
        return '\n'.join(lines)

    def sanity_check(self):
        assert self in self.subgraph.operators
        # check for duplicates
        assert len(self.inputs) == len(set(self.inputs))
        assert len(self.outputs) == len(set(self.outputs))
        # check double links with inputs/outputs
        for tensor in self.inputs:
            assert self in tensor.consumers
        for tensor in self.outputs:
            assert self in tensor.producers


class Tensor():
    def __init__(self, subgraph, name, type_, shape,
                 buffer=None, quantization=None,
                 producers=None, consumers=None):
        # Generally, do not use this constructor to instantiate Tensor!
        # Use Subgraph.create_tensor instead.
        self.subgraph = subgraph  # parent
        self.name = name
        assert isinstance(type_, TensorType)
        self.type = type_
        self.shape = list(shape)

        if buffer is None:
            self.buffer = self.model.create_buffer()
        else:
            assert isinstance(buffer, Buffer)
            assert buffer in self.model.buffers
            self.buffer = buffer
        self.buffer.owners.append(self)

        self.quantization = quantization

        self.producers = producers or []
        self.consumers = consumers or []

    @property
    def model(self):
        return self.subgraph.model

    def __str__(self):
        return f'name={self.name}, type={self.type.name}, shape={self.shape}, buffer={self.buffer}'

    def pprint(self):
        INDENT = ' ' * 2

        lines = []
        lines.append(str(self))
        if self.producers:
            lines.append(f'{INDENT}producers')
            lines.extend([f'{INDENT * 2}{producer}' for producer in self.producers])
        if self.consumers:
            lines.append(f'{INDENT}consumers')
            lines.extend([f'{INDENT * 2}{consumer}' for consumer in self.consumers])
        return '\n'.join(lines)

    def sanity_check(self):
        assert self in self.subgraph.tensors
        assert self in self.buffer.owners
        # check for duplicates
        assert len(self.consumers) == len(set(self.consumers))
        assert len(self.producers) == len(set(self.producers))
        # check double links with consumers/producers
        for op in self.producers:
            assert self in op.outputs
        for op in self.consumers:
            assert self in op.inputs

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
                      buffer=None, quantization=None,
                      isinput=False, isoutput=False,
                      producers=None, consumers=None):

        for existing_tensor in self.tensors:
            if name in [existing_tensor.name, existing_tensor.sanitized_name]:
                raise Exception(f'Tensor name {name} already in use')

        tensor = Tensor(self, name, type_, shape, buffer, quantization, producers, consumers)
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
        for op in tensor.consumers:
            try:
                op.inputs.remove(tensor)
            except ValueError:
                pass
        for op in tensor.producers:
            try:
                op.outputs.remove(tensor)
            except ValueError:
                pass
        tensor.consumers, tensor.producers = [], []
        tensor.buffer.owners.remove(tensor)
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
                        builtin_options=None, builtin_options_type=None,
                        custom_options=None):
        name = self.generate_unique_op_name(operator_code)
        operator = Operator(self, operator_code, name, inputs, outputs,
                            builtin_options, builtin_options_type, custom_options)
        self.operators.append(operator)
        for input_tensor in inputs:
            input_tensor.consumers.append(operator)
        for output_tensor in outputs:
            output_tensor.producers.append(operator)
        return operator

    def remove_operator(self, op):
        assert op in self.operators
        self.operators.remove(op)
        for t in op.inputs:
            try:
                t.consumers.remove(op)
            except ValueError:
                pass
        for t in op.outputs:
            try:
                t.producers.remove(op)
            except ValueError:
                pass
        op.inputs, op.outputs = [], []
        op.subgraph = None

    def insert_operator(self, ref_op, new_op, after=False):
        """NOTE: this does not rewire inputs/outputs"""
        # find location of reference op
        try:
            ref_idx = self.operators.index(ref_op)
        except ValueError as e:
            raise ValueError("Cannot find reference operator in the subgraph") from e

        # remove new_op from list if already in the subgraph
        if new_op in self.operators:
            self.operators.remove(new_op)

        # (re)insert new op after reference op
        self.operators.insert(ref_idx + (1 if after else 0), new_op)

    def replace_operator(self, op, new_op):
        """NOTE: this does not rewire inputs/outputs"""
        # insert new op
        try:
            self.insert_operator(op, new_op)
        except ValueError:
            raise ValueError("Cannot find operator to replace in the subgraph")
        # remove old op
        self.remove_operator(op)

    def get_tensor(self, name):
        for t in self.tensors:
            if t.name == name:
                return t
        raise ValueError(f"Tensor with name {name} not found!")

    def sanity_check(self):
        assert self in self.model.subgraphs
        # check for duplicates
        assert len(self.inputs) == len(set(self.inputs))
        assert len(self.outputs) == len(set(self.outputs))
        assert len(self.operators) == len(set(self.operators))
        assert len(self.tensors) == len(set(self.tensors))
        # make sure inputs and outputs are not misplaced
        for tensor in (self.inputs + self.outputs):
            assert tensor in self.tensors
        # the subgraph is sane as long as all its objects are sane
        for op in self.operators:
            op.sanity_check()
        for tensor in self.tensors:
            tensor.sanity_check()


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
        self.buffer.owners.append(self)

    def __str__(self):
        return f'name={self.name}, buffer={self.buffer}'

    def sanity_check(self):
        assert self in self.buffer.owners


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

    @property
    def data_size(self):
        nbytes = 0
        for buffer in self.buffers:
            nbytes += len(buffer)
        return nbytes

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
                print(operator.pprint())

            print('**********')
            print('* Inputs *')
            print('**********')
            for input_ in subgraph.inputs:
                print(input_.pprint())
                if tensor_values and len(input_.buffer):
                    print(f'   values={input_.numpy}')

            print('*****************')
            print('* Intermediates *')
            print('*****************')
            for intermediate in subgraph.intermediates:
                print(intermediate.pprint())
                if tensor_values and len(intermediate.buffer):
                    print(f'   values={intermediate.numpy}')

            print('***********')
            print('* Outputs *')
            print('***********')
            for output in subgraph.outputs:
                print(output.pprint())
                if tensor_values and len(output.buffer):
                    print(f'   values={output.numpy}')

    def sanity_check(self):
        # check for duplicates
        assert len(self.subgraphs) == len(set(self.subgraphs))
        assert len(self.buffers) == len(set(self.buffers))
        assert len(self.metadata) == len(set(self.metadata))
        # the model is sane as long as all its subgraphs are sane
        for subgraph in self.subgraphs:
            subgraph.sanity_check()
        for buffer in self.buffers:
            buffer.sanity_check()
        for metadata in self.metadata:
            metadata.sanity_check()
