# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import sys
import json
import tempfile
import ctypes

from .. import xcore_model
from .. import operator_codes

from . import flatbuffers_c

DEFAULT_SCHEMA = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'schema.fbs')

def create_buffer_from_dict(model, buffer_dict):
    if 'data' in buffer_dict:
        data = buffer_dict['data']
    else:
        data = []

    buffer = model.create_buffer(data)
    return buffer

def create_operator_from_dict(subgraph, tensors, operator_codes_dicts, operator_dict):
    inputs = []
    for input_index in operator_dict['inputs']:
        inputs.append(tensors[input_index])

    outputs = []
    for output_index in operator_dict['outputs']:
        outputs.append(tensors[output_index])

    operator_code_dict = operator_codes_dicts[operator_dict['opcode_index']]

    if operator_code_dict['builtin_code'] == 'CUSTOM':
        builtin_opcode = operator_codes.BuiltinOpCodes.CUSTOM
        custom_opcode = operator_codes.XCOREOpCodes(operator_code_dict['custom_code'])
    else:
        builtin_opcode = operator_codes.BuiltinOpCodes[operator_code_dict['builtin_code']]
        custom_opcode = None

    if 'builtin_options' in operator_dict:
        builtin_options = operator_dict['builtin_options']
    else:
        builtin_options = None

    if 'custom_options' in operator_dict:
        custom_options = operator_dict['custom_options']
    else:
        custom_options = None

    operator_code = operator_codes.OperatorCode(
        builtin_opcode,
        custom_code=custom_opcode,
        version=operator_code_dict['version']
    )

    operator = subgraph.create_operator(operator_code, inputs=inputs, outputs=outputs,
        builtin_options=builtin_options, custom_options=custom_options)

    return operator

def create_tensor_from_dict(subgraph, buffers, tensor_dict, is_input=False, is_output=False):
    name = tensor_dict['name']
    type_ = tensor_dict['type']
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


def read_flatbuffer(model_filename, schema=None):
    schema = schema or DEFAULT_SCHEMA

    parser = flatbuffers_c.FlatbufferIO()

    model_dict = json.loads(parser.read_flatbuffer(schema, model_filename))

    model = xcore_model.XCOREModel(
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

def write_flatbuffer(model_filename, schema=None):
    pass