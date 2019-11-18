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

class Operator():
    @classmethod
    def from_dict(cls, subgraph, operator_dict):
        '''Construct a Operator object from a TensorFlow Lite flatbuffer operator dictionary'''
        operator = cls()

        operator.subgraph = subgraph # parent

        operator.inputs = operator_dict['inputs']
        operator.outputs = operator_dict['outputs']
        if 'builtin_options' in operator_dict:
            operator.builtin_options = operator_dict['builtin_options']
        else:
            operator.builtin_options = None
        if 'custom_options' in operator_dict:
            operator.custom_options = operator_dict['custom_options']
        else:
            operator.custom_options = None
        operator.opcode_index = operator_dict['opcode_index']

        return operator

    def __str__(self):
        return f'inputs={self._inputs}, outputs={self._outputs}, opcode_index={self.opcode_index}'

class Tensor():
    @classmethod
    def from_dict(cls, subgraph, tensor_dict):
        '''Construct a Tensor object from a TensorFlow Lite flatbuffer tensor dictionary'''
        tensor = cls()

        tensor.subgraph = subgraph # parent

        tensor.name = tensor_dict['name']
        tensor.type = tensor_dict['type']
        tensor.shape = tensor_dict['shape']
        tensor.buffer = tensor_dict['buffer']
        if 'quantization' in tensor_dict:
            tensor.quantization = tensor_dict['quantization']
        else:
            tensor.quantization = None

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
        return self.GetNameSegments()[-1]

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
    def from_dict(cls, model, subgraph_dict):
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
            subgraph.tensors.append(Tensor.from_dict(subgraph, tensor_dict))

        # load operators
        subgraph.operators = []
        for operator_dict in subgraph_dict['operators']:
            subgraph.operators.append(Operator.from_dict(subgraph, operator_dict))

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

    def get_tensor(self, index):
        """Return one Tensor."""
        return self.tensors[index]

    def get_tensors(self, indices):
        """Return a list of Tensors with the given indices."""
        tensors = []

        for index in indices:
            tensors.append(self.tensors[index])

        return tensors

class XCOREModel():
    def __init__(self):
        self.buffers = None
        self.operator_codes = None
        self.subgraphs = None

    def Import(self, model_filename, flatc_bin=None, schema=None):
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

        # load buffers
        self.buffers = []
        for buffer in model['buffers']:
            if 'data' in buffer:
                self.buffers.append(buffer)
            else:
                self.buffers.append([])  # must append an empty so the indices match the buffer indiex in the tensors

        # load operator codes
        self.operator_codes = model['operator_codes']

        # load subgraphs
        self.subgraphs = []
        for subgraph in model['subgraphs']:
            self.subgraphs.append(Subgraph.from_dict(self, subgraph))

    def Export(self, model_filename):
        raise NotImplementedError #TODO: Implement me!!!

    def get_subgraph(self, index=0):
        return self.subgraphs[index]

    def get_buffer(self, index, stdtype = 'int8_t'):
        if 'data' in self.buffers[index]:
            bits = self.buffers[index]['data']
            if stdtype == 'int8_t':
                return bits
            elif stdtype == 'int16_t':
                return [i[0] for i in struct.iter_unpack('h', bytearray(bits))]
            elif stdtype == 'int32_t':
                return [i[0] for i in struct.iter_unpack('i', bytearray(bits))]

        return None
    
    def get_operator(self, index):
        if self.operator_codes[index]['builtin_code'] == 'CUSTOM':
            return self.operator_codes[index]['custom_code']
        else:
            return self.operator_codes[index]['builtin_code']
