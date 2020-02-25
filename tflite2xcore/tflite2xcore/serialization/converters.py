# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from .flatbuffers_c import FlexbufferBuilder


def create_dict_from_operator_code(operator_code):
    operator_code_dict = {
        'builtin_code': operator_code.builtin_code.name,
        'version': operator_code.version
    }

    if operator_code.custom_code:
        operator_code_dict['custom_code'] = operator_code.custom_code.name

    return operator_code_dict


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

    if operator.custom_options:
        fbb = FlexbufferBuilder(operator.custom_options)
        operator_dict['custom_options'] = fbb.get_bytes()

    return operator_dict


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


def create_dict_from_buffer(buffer):
    return {'data': buffer.data} if buffer.data is not None else {}


def create_dict_from_metadata(metadata):
    return {'name': metadata.name,
            'buffer': metadata.model.buffers.index(metadata.buffer)}


def create_dict_from_model(model):
    return {
        'version': model.version,
        'description': model.description,
        'metadata': [create_dict_from_metadata(mdata) for mdata in model.metadata],
        'buffers': [create_dict_from_buffer(buffer) for buffer in model.buffers],
        'subgraphs': [create_dict_from_subgraph(subgraph) for subgraph in model.subgraphs],
        'operator_codes': [create_dict_from_operator_code(operator_code) for operator_code in model.operator_codes]
    }
