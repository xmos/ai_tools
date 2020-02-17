# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import json
import re
import enum
import struct

import flatbuffers
import numpy as np

from .schema_py_generated import *
from .flatbuffers_c import FlexbufferBuilder, FlexbufferParser

from ..xcore_model import XCOREModel, TensorType
from ..operator_codes import OperatorCode, BuiltinOpCodes, XCOREOpCodes


class ActivationFunctionType(enum.Enum):
    NONE = 0
    RELU = 1
    RELU_N1_TO_1 = 2
    RELU6 = 3
    TANH = 4
    SIGN_BIT = 5

class QuantizationDetails(enum.Enum):
    NONE = 0
    CustomQuantization = 1

class Padding(enum.Enum):
    SAME = 0
    VALID = 1


# create enum at runtime for BuiltinOptions class in schema_py_generated
#    this is used for convenience
BuiltinOptionsEnum = enum.Enum(
    'BuiltinOptionsEnum',
    {k:v for k, v in vars(BuiltinOptions).items() if not k.startswith("__")}
)

def snake_to_camel(word):
    output = ''.join(x.capitalize() or '_' for x in word.split('_'))
    return output[0].lower() + output[1:]


def camel_to_snake(name):
  name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def builtin_options_to_dict(builtin_options):
    dict_ = {camel_to_snake(k):v for k, v in vars(builtin_options).items()}
    if 'fused_activation_function' in dict_:
        # convert enum value to string
        dict_['fused_activation_function'] = \
            ActivationFunctionType(dict_['fused_activation_function']).name
    if 'padding' in dict_:
        # convert enum value to string
        dict_['padding'] = \
            Padding(dict_['padding']).name

    return dict_


def dict_to_builtin_options(type_, dict_):
    class_identifier = BuiltinOptionsEnum(type_).name + 'T'

    builtin_class = globals()[class_identifier]
    builtin_options = builtin_class()

    for k, v in dict_.items():
        if k == 'fused_activation_function':
            # convert string to enum
            v = ActivationFunctionType[v].value
        elif k == 'padding':
            # convert string to enum
            v = Padding[v].value

        setattr(builtin_options, snake_to_camel(k), v)

    return builtin_options


def create_xcore_model(modelT):
    model = XCOREModel(
        version=modelT.version,
        description=modelT.description
    )

    # create buffers
    buffers = [model.create_buffer(**vars(bufferT)) for bufferT in modelT.buffers]

    # load metadata
    if modelT.metadata:
        for metadataT in modelT.metadata:
            model.create_metadata(metadataT.name, buffers[metadataT.buffer])

    # create operator codes lookup
    operator_codes_lut = []
    for operator_codeT in modelT.operatorCodes:
        if operator_codeT.builtinCode == BuiltinOpCodes.CUSTOM.value:
            opcode = XCOREOpCodes(operator_codeT.customCode.decode('utf-8'))
        else:
            opcode = BuiltinOpCodes(operator_codeT.builtinCode)
        operator_codes_lut.append(OperatorCode(opcode, version=operator_codeT.version))

    # load subgraphs
    for subgraphT in modelT.subgraphs:
        subgraph = model.create_subgraph(
            name=(subgraphT.name if hasattr(subgraphT, 'name') else None)
        )
        # load tensors
        tensors = []
        for tensor_index, tensorT in enumerate(subgraphT.tensors):
            is_input = tensor_index in subgraphT.inputs
            is_output = tensor_index in subgraphT.outputs

            # load quantization
            if hasattr(tensorT, 'quantization') and tensorT.quantization:
                quantization = {}
                for k, v in vars(tensorT.quantization).items():
                    if v is not None:
                        if k == 'details':
                            v = QuantizationDetails(v).name
                        elif isinstance(v, np.ndarray):
                            v = v.tolist()
                        quantization[camel_to_snake(k)] = v

            else:
                quantization = None

            tensor = subgraph.create_tensor(
                name=tensorT.name.decode('utf-8'),
                type_=TensorType(tensorT.type),
                shape=list(tensorT.shape.tolist() if tensorT.shape is not None  else []),
                buffer=buffers[tensorT.buffer],
                quantization=quantization,
                isinput=is_input,
                isoutput=is_output
            )
            tensors.append(tensor)

        # load operators
        for operatorT in subgraphT.operators:
            operator_code = operator_codes_lut[operatorT.opcodeIndex]
            options = {}
            if hasattr(operatorT, 'builtinOptions') and operatorT.builtinOptions is not None:
                options['builtin_options'] = builtin_options_to_dict(operatorT.builtinOptions)
                options['builtin_options_type'] = operatorT.builtinOptionsType

            if hasattr(operatorT, 'customOptions') and operatorT.customOptions is not None:
                options['custom_options'] = json.loads(
                    FlexbufferParser().parse(bytes(operatorT.customOptions))
                )

            subgraph.create_operator(
                operator_code,
                inputs=[tensors[input_index] for input_index in operatorT.inputs],
                outputs=[tensors[output_index] for output_index in operatorT.outputs],
                **options
            )

    return model

def create_flatbuffer_model(model):
    modelT = ModelT()
    modelT.version = model.version
    modelT.description = model.description

    # create buffers
    modelT.buffers = []
    for buffer in model.buffers:
        bufferT = BufferT()
        if len(buffer.data) > 0:
            bufferT.data = buffer.data
        modelT.buffers.append(bufferT)

    # create metadata
    modelT.metadata = []
    for metadata in model.metadata:
        metadataT = MetadataT()
        metadataT.name = metadata.name
        metadataT.buffer = model.buffers.index(metadata.buffer)
        modelT.metadata.append(metadataT)

    # create operator_codes
    modelT.operatorCodes = []
    for operator_code in model.operator_codes:
        operatorCodeT = OperatorCodeT()
        if operator_code.builtin_code:
            operatorCodeT.builtinCode = operator_code.builtin_code.value
        if operator_code.custom_code:
            operatorCodeT.customCode = operator_code.custom_code.name
        operatorCodeT.version = operator_code.version
        modelT.operatorCodes.append(operatorCodeT)

    # create subgraphs
    modelT.subgraphs = []
    for subgraph in model.subgraphs:
        subgraphT = SubGraphT()
        subgraphT.name = subgraph.name
        
        # set inputs
        subgraphT.inputs = []
        for input_ in subgraph.inputs:
            tensor_index = subgraph.tensors.index(input_)
            subgraphT.inputs.append(tensor_index)

        # set outputs
        subgraphT.outputs = []
        for output in subgraph.outputs:
            tensor_index = subgraph.tensors.index(output)
            subgraphT.outputs.append(tensor_index)

        # set tensors
        subgraphT.tensors = []
        for tensor in subgraph.tensors:
            tensorT = TensorT()
            tensorT.name = tensor.name
            tensorT.shape = tensor.shape
            tensorT.buffer = model.buffers.index(tensor.buffer)
            tensorT.type = tensor.type.value
            if tensor.quantization:
                quantizationT = QuantizationParametersT()
                if 'min' in tensor.quantization:
                    quantizationT.min = tensor.quantization['min']
                if 'max' in tensor.quantization:
                    quantizationT.max = tensor.quantization['max']
                if 'zero_point' in tensor.quantization:
                    quantizationT.zeroPoint = tensor.quantization['zero_point']
                if 'scale' in tensor.quantization:
                    quantizationT.scale = tensor.quantization['scale']
                if 'details_type' in tensor.quantization:
                    if isinstance(tensor.quantization['details_type'], str):
                        quantizationT.detailsType = QuantizationDetails[tensor.quantization['details_type']].value
                    else:
                        quantizationT.detailsType = tensor.quantization['details_type']
                if 'details' in tensor.quantization:
                    quantizationT.details = tensor.quantization['details']
                if 'quantized_dimension' in tensor.quantization:
                    quantizationT.quantizedDimension = tensor.quantization['quantized_dimension']
                tensorT.quantization = quantizationT
            subgraphT.tensors.append(tensorT)

        # set operators
        subgraphT.operators = []
        for operator in subgraph.operators:
            operatorT = OperatorT()
            operatorT.opcodeIndex = model.operator_codes.index(operator.operator_code)
            operatorT.inputs = []
            for input_tensor in operator.inputs:
                tensor_index = subgraph.tensors.index(input_tensor)
                operatorT.inputs.append(tensor_index)
            operatorT.outputs = []
            for output_tensor in operator.outputs:
                tensor_index = subgraph.tensors.index(output_tensor)
                operatorT.outputs.append(tensor_index)
            if operator.builtin_options:
                operatorT.builtinOptionsType = operator.builtin_options_type
                operatorT.builtinOptions = dict_to_builtin_options(operator.builtin_options_type, 
                    operator.builtin_options)
            if operator.custom_options:
                fbb = FlexbufferBuilder(operator.custom_options)
                operatorT.customOptions = fbb.get_bytes()
            subgraphT.operators.append(operatorT)

        modelT.subgraphs.append(subgraphT)

    return modelT


def read_flatbuffer(model_filename):
    with open(model_filename, "rb") as fd:
        bits = bytearray(fd.read())

    model_obj = Model.GetRootAsModel(bits, 0)
    modelT = ModelT.InitFromObj(model_obj)

    return create_xcore_model(modelT)


def write_flatbuffer(model, filename):
    modelT = create_flatbuffer_model(model)
    builder = flatbuffers.Builder(1024)
    model_offset = modelT.Pack(builder)

    builder.Finish(model_offset, file_identifier=b'TFL3')

    with open(filename, 'wb') as fd:
        return fd.write(builder.Output())

    return 0

