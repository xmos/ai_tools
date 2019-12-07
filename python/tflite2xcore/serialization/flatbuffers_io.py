# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import json

from .flatbuffers_c import FlexbufferBuilder, FlexbufferParser, FlatbufferIO

from ..xcore_model import XCOREModel, TensorType
from ..operator_codes import OperatorCode, BuiltinOpCodes, XCOREOpCodes

DEFAULT_SCHEMA = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'schema.fbs'
)


def create_buffer_from_dict(model, buffer_dict):
    # NOTE: if buffer_dict has a key other than 'data', this will fail
    # this is intentional, such buffer_dict should not exist
    return model.create_buffer(**buffer_dict)


def create_dict_from_buffer(buffer):
    return {'data': buffer.data} if buffer.data else {}


def create_operator_from_dict(subgraph, tensors, operator_code, operator_dict):
    options = {k: v for k, v in operator_dict.items()
               if k in ['builtin_options', 'builtin_options_type']}

    if 'custom_options' in operator_dict:
        options['custom_options'] = json.loads(
            FlexbufferParser().parse(bytes(operator_dict['custom_options']))
        )

    return subgraph.create_operator(
        operator_code,
        inputs=[tensors[input_index] for input_index in operator_dict['inputs']],
        outputs=[tensors[output_index] for output_index in operator_dict['outputs']],
        **options
    )


def create_dict_from_operator(operator):
    tensors = operator.subgraph.tensors
    operator_codes = operator.subgraph.model.operator_codes

    operator_dict = {
        'opcode_index': operator_codes.index(operator.operator_code),
        'inputs': [tensors.index(input_tensor) for input_tensor in operator.inputs],
        'outputs': [tensors.index(input_tensor) for input_tensor in operator.outputs],
        'custom_options_format': 'FLEXBUFFERS'
    }

    if operator.builtin_options:
        operator_dict['builtin_options'] = operator.builtin_options
        operator_dict['builtin_options_type'] = operator.builtin_options_type
    else:
        operator_dict['builtin_options_type'] = 'NONE'

    if operator.custom_options:
        fbb = FlexbufferBuilder(operator.custom_options)
        operator_dict['custom_options'] = fbb.get_bytes()

    return operator_dict


def create_operator_code_from_dict(operator_code_dict):
    if operator_code_dict['builtin_code'] == 'CUSTOM':
        opcode = XCOREOpCodes(operator_code_dict['custom_code'])
    else:
        opcode = BuiltinOpCodes[operator_code_dict['builtin_code']]

    return OperatorCode(opcode, version=operator_code_dict['version'])


def create_dict_from_operator_code(operator_code):
    operator_code_dict = {
        'builtin_code': operator_code.builtin_code.name,
        'version': operator_code.version
    }

    if operator_code.custom_code:
        operator_code_dict['custom_code'] = operator_code.custom_code.name

    return operator_code_dict


def create_tensor_from_dict(subgraph, buffers, tensor_dict, is_input=False, is_output=False):
    return subgraph.create_tensor(
        name=tensor_dict['name'],
        type_=TensorType[tensor_dict['type']],
        shape=tensor_dict['shape'],
        buffer=buffers[tensor_dict['buffer']],
        quantization=(tensor_dict['quantization'] if 'quantization' in tensor_dict else None),
        isinput=is_input,
        isoutput=is_output
    )


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


def create_subgraph_from_dict(model, buffers, operator_codes_dict, subgraph_dict):
    subgraph = model.create_subgraph(
        name=(subgraph_dict['name'] if 'name' in subgraph_dict else None)
    )

    # load tensors
    tensors = []
    for tensor_index, tensor_dict in enumerate(subgraph_dict['tensors']):
        is_input = tensor_index in subgraph_dict['inputs']
        is_output = tensor_index in subgraph_dict['outputs']
        tensor = create_tensor_from_dict(subgraph, buffers, tensor_dict, is_input, is_output)
        tensors.append(tensor)

    # create operator codes lookup
    operator_codes_lut = [create_operator_code_from_dict(operator_code_dict)
                          for operator_code_dict in operator_codes_dict]

    # load operators
    for operator_dict in subgraph_dict['operators']:
        operator_code = operator_codes_lut[operator_dict['opcode_index']]
        create_operator_from_dict(subgraph, tensors, operator_code, operator_dict)

    return subgraph


def create_dict_from_subgraph(subgraph):
    tensors = subgraph.tensors

    subgraph_dict = {
        'tensors': [create_dict_from_tensor(tensor) for tensor in tensors],
        'inputs': [tensors.index(input_tensor) for input_tensor in subgraph.inputs],
        'outputs': [tensors.index(output_tensor) for output_tensor in subgraph.outputs],
        'operators': [create_dict_from_operator(operator) for operator in subgraph.operators]
    }

    if subgraph.name:
        subgraph_dict['name'] = subgraph.name

    return subgraph_dict


def create_metadata_from_dict(model, buffers, metadata_dict):
    model.create_metadata(metadata_dict['name'],
                          buffers[metadata_dict['buffer']])


def create_dict_from_metadata(metadata):
    return {'name': metadata.name,
            'buffer': metadata.model.buffers.index(metadata.buffer)}


def read_flatbuffer(model_filename, schema=None):
    schema = schema or DEFAULT_SCHEMA

    parser = FlatbufferIO()

    model_dict = json.loads(parser.read_flatbuffer(schema, model_filename))

    model = XCOREModel(
        version=model_dict['version'],
        description=model_dict['description']
    )

    # create buffers
    buffers = [create_buffer_from_dict(model, buffer_dict)
               for buffer_dict in model_dict['buffers']]

    # load metadata
    for metadata_dict in model_dict['metadata']:
        create_metadata_from_dict(model, buffers, metadata_dict)

    # load subgraphs
    for subgraph_dict in model_dict['subgraphs']:
        create_subgraph_from_dict(model, buffers, model_dict['operator_codes'], subgraph_dict)

    return model


def write_flatbuffer(model, filename, schema=None):
    schema = schema or DEFAULT_SCHEMA

    model_dict = {
        'version': model.version,
        'description': model.description,
        'metadata': [create_dict_from_metadata(mdata) for mdata in model.metadata],
        'buffers': [create_dict_from_buffer(buffer) for buffer in model.buffers],
        'subgraphs': [create_dict_from_subgraph(subgraph) for subgraph in model.subgraphs],
        'operator_codes': [create_dict_from_operator_code(operator_code) for operator_code in model.operator_codes]
    }

    buffer = bytes(json.dumps(model_dict).encode('ascii'))
    builder = FlatbufferIO()
    bytes_written = builder.write_flatbuffer(schema, buffer, filename)

    return bytes_written
