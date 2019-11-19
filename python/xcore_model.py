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
    @classmethod
    def from_dict(cls, model, buffer_dict):
        buffer = cls()

        buffer.model = model # parent

        return buffer

class Operator():
    @classmethod
    def from_dict(cls, model, subgraph, operator_dict):
        '''Construct a Operator object from a TensorFlow Lite flatbuffer operator dictionary'''
        operator = cls()

        operator.model = model # grandparent
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
        #operator.opcode_index = operator_dict['opcode_index']

        return operator

    # def trim(self, cleanup_tensors=True):
    #     self.subgraph.trim_operator(self, cleanup_tensors)

    def __str__(self):
        return f'inputs={self.inputs}, outputs={self.outputs}, opcode_index={self.opcode_index}'

class Tensor():
    @classmethod
    def from_dict(cls, model, subgraph, tensor_dict):
        '''Construct a Tensor object from a TensorFlow Lite flatbuffer tensor dictionary'''
        tensor = cls()

        tensor.model = model # grandparent
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

    # def remove(self, cleanup_buffer=True):
    #     self.subgraph.remove_tensor(self, cleanup_buffer)

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
            subgraph.tensors.append(Tensor.from_dict(model, subgraph, tensor_dict))

        # load operators
        subgraph.operators = []
        for operator_dict in subgraph_dict['operators']:
            subgraph.operators.append(Operator.from_dict(model, subgraph, operator_dict))

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

    def get_tensors(self, indices):
        """Return a list of Tensors with the given indices."""
        tensors = []

        for index in indices:
            tensors.append(self.tensors[index])

        return tensors

    # def create_tensor(self, name, type_, shape, buffer=None, quantization=None):
    #     for existing_tensor in self.tensors:
    #         if name == existing_tensor.name:
    #             raise Exception(f'Tensor name {name} already in use')

    #     tensor = Tensor()
    #     tensor.subgraph = self
    #     tensor.name = name
    #     tensor.type = type_
    #     tensor.shape = shape
    #     tensor.buffer = buffer
    #     tensor.quantization = quantization

    #     return tensor

    # def remove_tensor(self, tensor, cleanup_buffer=True):
    #     # cleanup buffer
    #     if cleanup_buffer:
    #         self.model.cleanup_buffer(tensor.buffer)
        
    #     # fixup operator inputs and outputs
    #     tensor_index = self.tensors.index(tensor)
    #     for operator in self.operators:
    #         for list_index, input_index in enumerate(operator.inputs):
    #             if input_index == tensor_index:
    #                 operator.inputs.remove(input_index)
    #             elif input_index > tensor_index:
    #                 operator.inputs[list_index] -= 1
    #         for list_index, output_index in enumerate(operator.outputs):
    #             if input_index == tensor_index:
    #                 operator.outputs.remove(output_index)
    #             elif output_index > tensor_index:
    #                 operator.outputs[list_index] -= 1

    #     # fixup subgraph inputs and outputs
    #     if tensor in self.inputs:
    #         self.inputs.remove(tensor)
    #         # find new input
    #         first_operator = self.operators[0]
    #         new_input_tensor = self.tensors[first_operator.inputs[0]]
    #         self.inputs.append(new_input_tensor)
    #     elif tensor in self.outputs:
    #         self.outputs.remove(tensor)
    #         # find new output
    #         last_operator = self.operators[-1]
    #         new_output_tensor = self.tensors[last_operator.outputs[0]]
    #         self.outputs.append(new_output_tensor)

    #     # remove the tensor
    #     self.tensors.remove(tensor)

    # def create_operator(self, operator_index, inputs=None, outputs=None):
    #     for existing_operator in self.operators:
    #         if operator_index == existing_operator.opcode_index:
    #             return existing_operator

    #     operator_code = self.model.get_operator_code(operator_index)
        
    #     operator = Operator()
    #     operator.subgraph = self
    #     operator.inputs = inputs or []
    #     operator.outputs = outputs or []
    #     if 'builtin_options' in operator_code:
    #         operator.builtin_options = operator_code['builtin_options']
    #     else:
    #         operator.builtin_options = None
    #     if 'custom_options' in operator_code:
    #         operator.custom_options = operator_code['custom_options']
    #     else:
    #         operator.custom_options = None
    #     operator.opcode_index = operator_index

    #     return operator

    # def trim_operator(self, operator, cleanup_tensors=True):
    #     operator_index = self.operators.index(operator)
    #     first_operator = operator_index == 0
    #     last_operator = operator_index == len(self.operators) - 1
    #     if not first_operator and not last_operator:
    #         raise Exception('Only first or last operator can be trimmed')
    #     # remove the operator
    #     #   NOTE: do this first, tensor cleanup depends on it!
    #     self.operators.remove(operator)

    #     # cleanup input and output tensors
    #     if cleanup_tensors:
    #         if first_operator:
    #             for tensor_index in operator.inputs:
    #                 self.remove_tensor(self.tensors[tensor_index], True)
    #         if last_operator:
    #             for tensor_index in operator.outputs:
    #                 self.remove_tensor(self.tensors[tensor_index], True)

    #     # cleanup opcode
    #     self.model.cleanup_operator_code(operator.opcode_index)

class XCOREModel():
    def __init__(self):
        self.buffers = None
        self.operator_codes = None
        self.subgraphs = None

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

    def save(self, model_filename):
        raise NotImplementedError #TODO: Implement me!!!

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
    
    # def create_buffer(self, data=None):
    #     if data:
    #         self.buffers.append({'data': data})
    #     else:
    #         self.buffers.append([])

    #     return len(self.buffers) - 1

    # def cleanup_buffer(self, index):
    #     # Removes a buffer if only 1 reference to it exists
        
    #     # compute reference count
    #     reference_count = 0
    #     for subgraph in self.subgraphs:
    #         for tensor in subgraph.tensors:
    #             if tensor.buffer == index:
    #                 reference_count += 1

    #     if reference_count <=1:
    #         del self.buffers[index]

    #         # fixup tensors
    #         for subgraph in self.subgraphs:
    #             for tensor in subgraph.tensors:
    #                 if tensor.buffer > index:
    #                     tensor.buffer -= 1

    def get_operator_code(self, index):
        if self.operator_codes[index]['builtin_code'] == 'CUSTOM':
            return self.operator_codes[index]['custom_code']
        else:
            return self.operator_codes[index]['builtin_code']

    # def create_operator_code(self, builtin_code=None, custom_code=None, version=1):
    #     assert(builtin_code or custom_code)

    #     # first search for custom_code
    #     for index, operator_code in enumerate(self.operator_codes):
    #         if builtin_code and 'builtin_code' in operator_code:
    #             if operator_code['builtin_code'] == builtin_code:
    #                 return index
    #         elif custom_code and 'custom_code' in operator_code:
    #             if operator_code['custom_code'] == custom_code:
    #                 return index

    #     # now create
    #     if builtin_code:
    #         operator_code = {'builtin_code': builtin_code, 'custom_code': None, 'version': version}
    #     elif custom_code:
    #         operator_code = {'builtin_code': 'CUSTOM', 'custom_code': custom_code, 'version': version}

    #     self.operator_codes.append(operator_code)
    #     return len(self.operator_codes) - 1

    # def cleanup_operator_code(self, index):
    #     # Removes an operator if only 1 reference to it exists

    #     # compute reference count
    #     reference_count = 0
    #     for subgraph in self.subgraphs:
    #         for operator in subgraph.operators:
    #             if operator.opcode_index == index:
    #                 reference_count += 1

    #     if reference_count <=1:
    #         del self.operator_codes[index]

    #         # fixup operators
    #         for subgraph in self.subgraphs:
    #             for operator in subgraph.operators:
    #                 if operator.opcode_index > index:
    #                     operator.opcode_index -= 1

