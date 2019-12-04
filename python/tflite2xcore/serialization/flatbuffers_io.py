# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import sys
import json
import tempfile
import ctypes

from .flatbuffers_c import FlexbufferBuilder, FlexbufferParser, FlatbufferIO

from xcore_model import XCOREModel, TensorType
from operator_codes import OperatorCode, BuiltinOpCodes, XCOREOpCodes

DEFAULT_SCHEMA = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'schema.fbs')

def create_buffer_from_dict(model, buffer_dict):
    if 'data' in buffer_dict:
        data = buffer_dict['data']
    else:
        data = []

    buffer = model.create_buffer(data)
    return buffer

def create_dict_from_buffer(buffer):
    if buffer.data:
        return {'data': buffer.data}
    else:
        return {}

def create_operator_from_dict(subgraph, tensors, operator_codes_dicts, operator_dict):
    inputs = []
    for input_index in operator_dict['inputs']:
        inputs.append(tensors[input_index])

    outputs = []
    for output_index in operator_dict['outputs']:
        outputs.append(tensors[output_index])

    operator_code_dict = operator_codes_dicts[operator_dict['opcode_index']]

    if operator_code_dict['builtin_code'] == 'CUSTOM':
        builtin_opcode = BuiltinOpCodes.CUSTOM
        custom_opcode = XCOREOpCodes(operator_code_dict['custom_code'])
    else:
        builtin_opcode = BuiltinOpCodes[operator_code_dict['builtin_code']]
        custom_opcode = None

    if 'builtin_options' in operator_dict:
        builtin_options = operator_dict['builtin_options']
        builtin_options_type = operator_dict['builtin_options_type']
    else:
        builtin_options = None
        builtin_options_type = None

    if 'custom_options' in operator_dict:
        # custom_options = bytes(operator_dict['custom_options'])
        parser = FlexbufferParser()
        custom_options = json.loads(parser.parse(bytes(operator_dict['custom_options'])))
    else:
        custom_options = None

    operator_code = OperatorCode(
        builtin_opcode,
        custom_code=custom_opcode,
        version=operator_code_dict['version']
    )

    operator = subgraph.create_operator(operator_code, inputs=inputs, outputs=outputs,
        builtin_options=builtin_options, builtin_options_type=builtin_options_type, 
        custom_options=custom_options)

    return operator

def create_dict_from_operator(operator):
    tensors = operator.subgraph.tensors
    operator_codes = operator.subgraph.model.operator_codes

    operator_dict = {
        'opcode_index': operator_codes.index(operator.operator_code),
        'inputs': [],
        'outputs': [],
        'custom_options_format': 'FLEXBUFFERS'
    }

    for input_tensor in operator.inputs:
        tensor_index = tensors.index(input_tensor)
        operator_dict['inputs'].append(tensor_index)

    for input_tensor in operator.outputs:
        tensor_index = tensors.index(input_tensor)
        operator_dict['outputs'].append(tensor_index)

    if operator.builtin_options:
        operator_dict['builtin_options'] = operator.builtin_options
        operator_dict['builtin_options_type'] = operator.builtin_options_type
    else:
        operator_dict['builtin_options_type'] = 'NONE'

    if operator.custom_options:
        fbb = FlexbufferBuilder(operator.custom_options)
        operator_dict['custom_options'] = fbb.get_bytes()

    return operator_dict

def create_dict_from_operator_code(operator_code):
    operator_code_dict = {
        'builtin_code': operator_code.builtin_code.name,
        'version': operator_code.version
    }

    if operator_code.custom_code:
        operator_code_dict['custom_code'] = operator_code.custom_code.name

    return operator_code_dict

def create_tensor_from_dict(subgraph, buffers, tensor_dict, is_input=False, is_output=False):
    name = tensor_dict['name']
    type_ = TensorType[tensor_dict['type']]
    shape = tensor_dict['shape']
    buffer = buffers[tensor_dict['buffer']]
    if 'quantization' in tensor_dict:
        quantization = tensor_dict['quantization']
    else:
        quantization = None

    tensor = subgraph.create_tensor(name, type_, shape,
        buffer=buffer, quantization=quantization,
        isinput=is_input, isoutput=is_output)

    return tensor

def create_dict_from_tensor(tensor):
    buffers = tensor.subgraph.model.buffers

    tensor_dict = {
        'name': tensor.name,
        'type': tensor.type.name,
        'shape': tensor.shape,
        'buffer': buffers.index(tensor.buffer)
    }

    if tensor.quantization:
        tensor_dict['quantization'] = tensor.quantization

    return tensor_dict

def create_subgraph_from_dict(model, buffers, operator_codes, subgraph_dict):
    subgraph = model.create_subgraph()

    if 'name' in subgraph_dict:
        subgraph.name = subgraph_dict['name']
    else:
        subgraph.name = None

    # load tensors
    tensors = []
    for tensor_index, tensor_dict in enumerate(subgraph_dict['tensors']):
        is_input = tensor_index in subgraph_dict['inputs']
        is_output = tensor_index in subgraph_dict['outputs']
        tensor = create_tensor_from_dict(subgraph, buffers, tensor_dict, is_input, is_output)
        tensors.append(tensor)

    # load operators
    for operator_dict in subgraph_dict['operators']:
        create_operator_from_dict(subgraph, tensors, operator_codes, operator_dict)

    return subgraph

def create_dict_from_subgraph(subgraph):
    subgraph_dict = {}

    if subgraph.name:
        subgraph_dict['name'] = subgraph.name

    # tensors
    tensors = subgraph.tensors
    subgraph_dict['tensors'] = []
    for tensor_index, tensor in enumerate(tensors):
        tensor_dict = create_dict_from_tensor(tensor)
        subgraph_dict['tensors'].append(tensor_dict)

    # inputs & outputs
    subgraph_dict['inputs'] = []
    for input_tensor in subgraph.inputs:
        tensor_index = tensors.index(input_tensor)
        subgraph_dict['inputs'].append(tensor_index)

    subgraph_dict['outputs'] = []
    for output_tensor in subgraph.outputs:
        tensor_index = tensors.index(output_tensor)
        subgraph_dict['outputs'].append(tensor_index)

    # operators
    subgraph_dict['operators'] = []
    for operator in subgraph.operators:
        operator_dict = create_dict_from_operator(operator)
        subgraph_dict['operators'].append(operator_dict)

    return subgraph_dict

def read_flatbuffer(model_filename, schema=None):
    schema = schema or DEFAULT_SCHEMA

    parser = FlatbufferIO()

    model_dict = json.loads(parser.read_flatbuffer(schema, model_filename))

    model = XCOREModel(
        version = model_dict['version'],
        description = model_dict['description'],
        metadata = model_dict['metadata']
    )

    # create buffers
    buffers = []
    for buffer_dict in model_dict['buffers']:
        buffer = create_buffer_from_dict(model, buffer_dict)
        buffers.append(buffer)

    # load subgraphs
    for subgraph_dict in model_dict['subgraphs']:
        create_subgraph_from_dict(model, buffers, model_dict['operator_codes'], subgraph_dict)

    return model

def write_flatbuffer(model, filename, schema=None):
    schema = schema or DEFAULT_SCHEMA

    model_dict = {
        'version': model.version,
        'description': model.description,
        'metadata': model.metadata,
        'buffers': [],
        'subgraphs': [],
        'operator_codes': []
    }

    # buffers
    for buffer in model.buffers:
        buffer_dict = create_dict_from_buffer(buffer)
        model_dict['buffers'].append(buffer_dict)

    # subgraphs
    for subgraph in model.subgraphs:
        subgraph_dict = create_dict_from_subgraph(subgraph)
        model_dict['subgraphs'].append(subgraph_dict)

    # operator codes
    for operator_code in model.operator_codes:
        operator_code_dict = create_dict_from_operator_code(operator_code)
        model_dict['operator_codes'].append(operator_code_dict)

    buffer = bytes(json.dumps(model_dict).encode('ascii'))
    builder = FlatbufferIO()
    bytes_written = builder.write_flatbuffer(schema, buffer, filename)

    return bytes_written
