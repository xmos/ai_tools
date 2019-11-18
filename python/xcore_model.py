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
    def from_dict(cls, operator_dict):
        '''Construct a Operator object from a TensorFlow Lite flatbuffer operator dictionary'''
        operator = cls()

        operator.inputs = operator_dict['inputs']
        operator.outputs = operator_dict['outputs']
        #operator.builtin_options = tensor_dict['builtin_options']
        operator.opcode_index = operator_dict['opcode_index']

        return operator

    def __str__(self):
        return f'inputs={self.inputs}, outputs={self.outputs}, opcode_index={self.opcode_index}'

    def GetInputs(self):
        return self.inputs

    def GetOutputs(self):
        return self.outputs

    def GetOpcodeIndex(self):
        return self.opcode_index

class Tensor():
    @classmethod
    def from_dict(cls, tensor_dict):
        '''Construct a Tensor object from a TensorFlow Lite flatbuffer tensor dictionary'''
        tensor = cls()

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

    def GetName(self):
        return self.name

    def GetSanitizedName(self):
        '''Return a name that is safe to use in source code'''
        return self.name.replace('/', '_')

    def GetName(self):
        '''Return a name that is safe to use in source code'''
        return self.name.replace('/', '_')

    def GetNameSegments(self):
        return self.name.split('/')

    def GetBaseName(self):
        return self.GetNameSegments()[-1]

    def GetType(self):
        return self.type

    def GetStandardType(self):
        '''Return type (from cstdint.h)'''
        return TFLITE_TYPE_TO_C_TYPE[self.type]

    def GetShape(self):
        return self.shape

    def GetSize(self):
        size = TFLITE_TYPE_TO_BYTES[self.type]
        for s in self.shape:
            size *= s
        return size

    def GetBuffer(self):
        return self.buffer

    def GetQuantization(self):
        return self.quantization

class Subgraph():
    @classmethod
    def from_dict(cls, subgraph_dict):
        '''Construct a Subgraph object from a TensorFlow Lite flatbuffer subgraph dictionary'''
        subgraph = cls()

        if 'name' in subgraph_dict:
            subgraph.name = subgraph_dict['name']
        else:
            subgraph.name = None

        # load tensors
        subgraph.tensors = []
        for tensor_dict in subgraph_dict['tensors']:
            subgraph.tensors.append(Tensor.from_dict(tensor_dict))

        # load operators
        subgraph.operators = []
        for operator_dict in subgraph_dict['operators']:
            subgraph.operators.append(Operator.from_dict(operator_dict))

        input_output_indices = set([]) # lookup for input and output tensor indices
        # load inputs
        subgraph.inputs = []
        for input_index in subgraph_dict['inputs']:
            input_output_indices.add(input_index)
            subgraph.inputs.append(subgraph.tensors[input_index])

        # load outputs
        subgraph.outputs = []
        for output_index in subgraph_dict['outputs']:
            input_output_indices.add(output_index)
            subgraph.outputs.append(subgraph.tensors[output_index])

        # load intermediates
        #   intermediates are any tensors that are not an input or an output
        subgraph.intermediates = []
        for tensor_index, tensor in enumerate(subgraph.tensors):
            if tensor_index not in input_output_indices:
                subgraph.intermediates.append(tensor)

        return subgraph

    def GetName(self):
        """Return name."""
        return self.name

    def GetTensors(self):
        """Return all Tensors."""
        return self.tensors

    def GetTensor(self, index):
        """Return one Tensor."""
        return self.tensors[index]

    def GetTensors(self, indices):
        """Return a list of Tensors with the given indices."""
        tensors = []

        for index in indices:
            tensors.append(self.tensors[index])

        return tensors

    def GetOperators(self):
        """Return all Operators."""
        return self.operators

    def GetInputs(self):
        """Return the input Tensors."""
        return self.inputs

    def GetOutputs(self):
        """Return the output Tensors."""
        return self.outputs

    def GetInitializers(self):
        """Return the initialized Tensors."""
        return self.initializers

    def GetIntermediates(self):
        """Return the intermediate Tensors."""
        return self.intermediates

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
            self.subgraphs.append(Subgraph.from_dict(subgraph))

    def Export(self, model_filename):
        raise NotImplementedError #TODO: Implement me!!!

    def GetSubgraph(self, index=0):
        return self.subgraphs[index]

    def GetBuffers(self):
        return self.buffers

    def GetBuffer(self, index, stdtype = 'int8_t'):
        if 'data' in self.buffers[index]:
            bits = self.buffers[index]['data']
            if stdtype == 'int8_t':
                return bits
            elif stdtype == 'int16_t':
                return [i[0] for i in struct.iter_unpack('h', bytearray(bits))]
            elif stdtype == 'int32_t':
                return [i[0] for i in struct.iter_unpack('i', bytearray(bits))]

        return None
    
    def GetOperator(self, index):
        if self.operator_codes[index]['builtin_code'] == 'CUSTOM':
            return self.operator_codes[index]['custom_code']
        else:
            return self.operator_codes[index]['builtin_code']
