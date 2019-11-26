# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import sys
import json
import tempfile
import ctypes

from .. import xcore_model
from .. import OperatorCodes

from . import flatbuffers_c

# def __norm_and_join(*args):
#     return os.path.normpath(os.path.join(*args))

# __io_dir = __norm_and_join(
#     os.path.dirname(os.path.realpath(__file__)), '')

# DEFAULT_SCHEMA = __norm_and_join(__flatbuffer_xmos_dir, 'schema.fbs')

# DEFAULT_SCHEMA = os.path.normpath(os.path.join(*args)) os.path.realpath(__file__, 'schema.fbs')


DEFAULT_SCHEMA = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'schema.fbs')
# def load_tflite_as_json(tflite_input, *,
#                         flatc_bin=DEFAULT_FLATC, schema=DEFAULT_SCHEMA):
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         # convert model to json
#         cmd = (f"{flatc_bin} -t --strict-json --defaults-json "
#                f"-o {tmp_dir} {schema} -- {tflite_input}")
#         os.system(cmd)

#         # open json file
#         json_input = os.path.join(
#             tmp_dir,
#             os.path.splitext(os.path.basename(tflite_input))[0] + ".json")
#         with open(json_input, 'r') as f:
#             model = json.load(f)

#     return model

# def save_json_as_tflite(model, tflite_output, *,
#                         flatc_bin=DEFAULT_FLATC, schema=DEFAULT_SCHEMA):
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         # write new json file
#         json_output = os.path.join(
#             tmp_dir,
#             os.path.splitext(os.path.basename(tflite_output))[0] + ".tmp.json")
#         with open(json_output, 'w') as f:
#             json.dump(model, f, indent=2)

#         # convert to tflite
#         cmd = (f"{flatc_bin} -b --strict-json --defaults-json "
#                f"-o {tmp_dir} {schema} {json_output}")
#         logging.info(f"Executing: {cmd}")
#         os.system(cmd)

#         # move to specified location
#         tmp_tflite_output = os.path.join(
#             tmp_dir,
#             os.path.splitext(os.path.basename(tflite_output))[0] + ".tmp.tflite")
#         shutil.move(tmp_tflite_output, tflite_output)

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
        builtin_opcode = OperatorCodes.BuiltinOpCodes.CUSTOM
        custom_opcode = OperatorCodes.XCOREOpCodes(operator_code_dict['custom_code'])
    else:
        builtin_opcode = OperatorCodes.BuiltinOpCodes[operator_code_dict['builtin_code']]
        custom_opcode = None

    if 'builtin_options' in operator_dict:
        builtin_options = operator_dict['builtin_options']
    else:
        builtin_options = None

    if 'custom_options' in operator_dict:
        custom_options = operator_dict['custom_options']
    else:
        custom_options = None

    operator_code = OperatorCodes.OperatorCode(
        builtin_opcode,
        custom_opcode=custom_opcode,
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


def read_flatbuffers_json(model_filename, schema=None):
    schema = schema or DEFAULT_SCHEMA

    parser = flatbuffers_c.FlatbufferParser()

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

# def old_read_flatbuffers_json(model_filename, flatc_bin=None, schema=None):
#     #TODO: replace with idea from https://github.com/google/flatbuffers/issues/4403
#     flatc_bin = flatc_bin or DEFAULT_FLATC
#     schema = schema or DEFAULT_SCHEMA
    
#     # loads by first converting to json (yuck!)
#     if not os.path.exists(schema):
#         raise FileNotFoundError(
#             'Schema file cannot be found at {schema}')

#     if flatc_bin is None:
#         raise RuntimeError('Cannot find flatc')
#     elif not os.path.exists(flatc_bin):
#         raise RuntimeError(
#             f'Flatc is not available at {flatc_bin}')

#     model_dict = load_tflite_as_json(model_filename,
#                                 flatc_bin=flatc_bin, schema=schema)

#     model = xcore_model.XCOREModel(
#         version = model_dict['version'],
#         description = model_dict['description'],
#         metadata = model_dict['metadata']
#     )

#     # create buffers
#     buffers = []
#     for buffer_dict in model_dict['buffers']:
#         buffer = create_buffer_from_dict(model, buffer_dict)
#         buffers.append(buffer)

#     # load subgraphs
#     for subgraph_dict in model_dict['subgraphs']:
#         create_subgraph_from_dict(model, buffers, model_dict['operator_codes'], subgraph_dict)

#     return model