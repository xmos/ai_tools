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


def create_dict_from_tensor(tensor, *, extended=False):
    subgraph = tensor.subgraph
    buffers = subgraph.model.buffers

    tensor_dict = {
        'name': tensor.name,
        'type': tensor.type.name,
        'shape': tensor.shape,
        'buffer': buffers.index(tensor.buffer)
    }

    if tensor.quantization:
        tensor_dict['quantization'] = tensor.quantization

    if extended:
        operators = subgraph.operators
        tensor_dict['consumers'] = sorted(operators.index(t)
                                          for t in tensor.consumers)
        tensor_dict['producers'] = sorted(operators.index(t)
                                          for t in tensor.producers)

    return tensor_dict


def create_dict_from_subgraph(subgraph, *, extended=False):
    tensors = subgraph.tensors

    subgraph_dict = {
        'tensors': [create_dict_from_tensor(tensor, extended=extended)
                    for tensor in tensors],
        'inputs': [tensors.index(input_tensor) for input_tensor in subgraph.inputs],
        'outputs': [tensors.index(output_tensor) for output_tensor in subgraph.outputs],
        'operators': [create_dict_from_operator(operator) for operator in subgraph.operators]
    }

    if subgraph.name:
        subgraph_dict['name'] = subgraph.name

    return subgraph_dict


def create_dict_from_buffer(buffer, *, extended=False):
    buffer_dict = {'data': buffer.data} if buffer.data is not None else {}

    if extended:
        owners_dict = dict()
        model = buffer.model

        # track down and tally all owners
        for owner in buffer.owners:
            if owner in model.metadata:
                metadata_owners = owners_dict.setdefault('metadata', [])
                metadata_owners.append(owner.name)
            else:  # owner is a tensor
                subgraph = owner.subgraph
                subgraph_idx = model.subgraphs.index(subgraph)
                owners_in_subgraph = owners_dict.setdefault(subgraph_idx, [])
                owners_in_subgraph.append(subgraph.tensors.index(owner))

        # sort the ordering
        owners_dict = dict(sorted(owners_dict.items()))
        for subgraph_idx in owners_dict:
            owners_dict[subgraph_idx].sort()

        buffer_dict['owners'] = owners_dict

    return buffer_dict


def create_dict_from_metadata(metadata):
    return {'name': metadata.name,
            'buffer': metadata.model.buffers.index(metadata.buffer)}


def create_dict_from_model(model, *, extended=False):
    return {
        'version': model.version,
        'description': model.description,
        'metadata': [create_dict_from_metadata(metadata)
                     for metadata in model.metadata],
        'buffers': [create_dict_from_buffer(buffer, extended=extended)
                    for buffer in model.buffers],
        'subgraphs': [create_dict_from_subgraph(subgraph, extended=extended)
                      for subgraph in model.subgraphs],
        'operator_codes': [create_dict_from_operator_code(operator_code)
                           for operator_code in model.operator_codes]
    }
