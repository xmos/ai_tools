# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import struct

from tflite_utils import DEFAULT_FLATC, DEFAULT_SCHEMA
from tflite_utils import load_tflite_as_json

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
        self.model = model # parent
        self.data = data or []

    @classmethod
    def from_dict(cls, model, buffer_dict):
        if 'data' in buffer_dict:
            data = buffer_dict['data']
        else:
            data = []

        buffer = cls(model, data)

        return buffer

    def __str__(self):
        if self.data:
            len_ = len(self.data)
            return f'{len_}'
        else:
            return f'[]'

    def unpack(self, stdtype='int8_t'):
        if stdtype == 'int8_t':
            return self.data
        elif stdtype == 'int16_t':
            return [i[0] for i in struct.iter_unpack('h', bytearray(self.data))]
        elif stdtype == 'int32_t':
            return [i[0] for i in struct.iter_unpack('i', bytearray(self.data))]

class Operator():
    def __init__(self, model, subgraph, operator_code, inputs=None, 
                 outputs=None, builtin_options=None, custom_options=None):
        self.model = model # grandparent
        self.subgraph = subgraph # parent
        self.operator_code = operator_code
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.builtin_options = builtin_options
        self.custom_options = custom_options

    @classmethod
    def from_dict(cls, model, subgraph, operator_codes, operator_dict):
        '''Construct a Operator object from a TensorFlow Lite flatbuffer operator dictionary'''
        inputs = []
        for input_index in operator_dict['inputs']:
            input_tensor = subgraph.tensors[input_index]
            inputs.append(input_tensor)

        outputs = []
        for output_index in operator_dict['outputs']:
            output_tensor = subgraph.tensors[output_index]
            outputs.append(output_tensor)

        if 'builtin_options' in operator_dict:
            builtin_options = operator_dict['builtin_options']
        else:
            builtin_options = None
        if 'custom_options' in operator_dict:
            custom_options = operator_dict['custom_options']
        else:
            custom_options = None

        operator_code = operator_codes[operator_dict['opcode_index']]

        operator = cls(model, subgraph, operator_code, inputs, outputs, builtin_options, custom_options)

        return operator

    def __str__(self):
        INDENT='  '

        lines = []
        lines.append(f'operator_code= {self.operator_code}')
        lines.append(INDENT+'inputs')
        lines.extend([INDENT+INDENT+str(input_) for input_ in self.inputs])
        lines.append(INDENT+'outputs')
        lines.extend([INDENT+INDENT+str(output) for output in self.outputs])
        return '\n'.join(lines)

class Tensor():
    def __init__(self, model, subgraph, name, type_, shape, buffer=None, quantization=None):
        self.model = model # grandparent
        self.subgraph = subgraph # parent
        self.name = name
        self.type = type_
        self.shape = shape
        if buffer:
            self.buffer = buffer
        else:
            self.buffer = Buffer(self.model)
            self.model.buffers.append(self.buffer)
        self.quantization = quantization

    @classmethod
    def from_dict(cls, model, subgraph, tensor_dict):
        '''Construct a Tensor object from a TensorFlow Lite flatbuffer tensor dictionary'''
        name = tensor_dict['name']
        type_ = tensor_dict['type']
        shape = tensor_dict['shape']
        buffer = model.buffers[tensor_dict['buffer']]
        if 'quantization' in tensor_dict:
            quantization = tensor_dict['quantization']
        else:
            quantization = None

        tensor = cls(model, subgraph, name, type_, shape, buffer, quantization)

        return tensor

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
    @classmethod
    def from_dict(cls, model, operator_codes, subgraph_dict):
        '''Construct a Subgraph object from a TensorFlow Lite flatbuffer subgraph dictionary'''
        subgraph = cls()

        subgraph.model = model # parent

        if 'name' in subgraph_dict:
            subgraph.name = subgraph_dict['name']
        else:
            subgraph.name = None

        # load tensors
        subgraph.tensors = []
        for tensor_dict in subgraph_dict['tensors']:
            subgraph.tensors.append(Tensor.from_dict(model, subgraph, tensor_dict))

        # load operators
        subgraph.operators = []
        for operator_dict in subgraph_dict['operators']:
            subgraph.operators.append(Operator.from_dict(model, subgraph, operator_codes, operator_dict))

        # load inputs
        subgraph.inputs = []
        for input_index in subgraph_dict['inputs']:
            subgraph.inputs.append(subgraph.tensors[input_index])

        # load outputs
        subgraph.outputs = []
        for output_index in subgraph_dict['outputs']:
            subgraph.outputs.append(subgraph.tensors[output_index])

        return subgraph

    @property
    def intermediates(self):
        #   intermediates are any tensors that are not an input or an output
        input_output_tensors = set([]) # lookup for input and output tensor indices
        for input_ in self.inputs:
            input_output_tensors.add(input_.name)
        for output in self.outputs:
            input_output_tensors.add(output.name)

        intermediates = []
        for tensor, tensor in enumerate(self.tensors):
            if tensor.name not in input_output_tensors:
                intermediates.append(tensor)

        return intermediates

    def create_tensor(self, name, type_, shape, buffer=None, quantization=None):
        for existing_tensor in self.tensors:
            if name == existing_tensor.name:
                raise Exception(f'Tensor name {name} already in use')

        tensor = Tensor(self.model, self, name, type_, shape, buffer, quantization)
        self.tensors.append(tensor)

        return tensor

    def create_operator(self, operator_code, inputs=None, outputs=None, builtin_options=None, custom_options=None):
        operator = Operator(self.model, self, operator_code, inputs, outputs, builtin_options, custom_options)
        self.operators.append(operator)

        return operator

class XCOREModel():
    def __init__(self, version=None, description=None, subgraphs=None, buffers=None, metadata=None):
        self.version = version
        self.description = description
        self.buffers = buffers or []
        self.subgraphs = subgraphs or []
        self.metadata = metadata

    def load(self, model_filename, flatc_bin=None, schema=None):
        #TODO: replace with idea from https://github.com/google/flatbuffers/issues/4403
        flatc_bin = flatc_bin or DEFAULT_FLATC
        schema = schema or DEFAULT_SCHEMA
        
        # loads by first converting to json (yuck!)
        if not os.path.exists(schema):
            raise FileNotFoundError(
                "Sorry, schema file cannot be found at {}".format(schema))

        if flatc_bin is None:
            raise RuntimeError("Sorry, cannot find flatc")
        elif not os.path.exists(flatc_bin):
            raise RuntimeError(
                "Sorry, flatc is not available at {}".format(flatc_bin))

        model = load_tflite_as_json(model_filename,
                                    flatc_bin=flatc_bin, schema=schema)

        self.version = model['version']
        self.description = model['description']

        # load buffers
        self.buffers = []
        for buffer in model['buffers']:
            self.buffers.append(Buffer.from_dict(self, buffer))

        # load subgraphs
        self.subgraphs = []
        for subgraph in model['subgraphs']:
            self.subgraphs.append(Subgraph.from_dict(self, model['operator_codes'], subgraph))

        self.metadata = model['metadata']

    def save(self, model_filename):
        raise NotImplementedError #TODO: Implement me!!!

    def create_buffer(self, data=None):
        buffer = Buffer(self, data)
        self.buffers.append(buffer)
        return buffer

    @property
    def operator_codes(self):
        operator_codes = {}

        for subgraph in self.subgraphs:
            for operator in subgraph.operators:
                print(operator)
                operator_codes[str(operator.operator_code)] = operator.operator_code

        return operator_codes.values()

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

