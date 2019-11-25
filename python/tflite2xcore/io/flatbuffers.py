# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import sys
import re
import importlib

from .schema_generated import Model, BuiltinOptions
from .. import xcore_model
from .. import OperatorCodes

# load the builtin_options class types
BUILTINOPTIONS_TYPE_NAMES = [a for a in dir(BuiltinOptions.BuiltinOptions()) if not a.startswith('__')]
BUILTINOPTIONS_TYPES = [None] * len(BUILTINOPTIONS_TYPE_NAMES)
for n in BUILTINOPTIONS_TYPE_NAMES:
    BUILTINOPTIONS_TYPES[getattr(BuiltinOptions.BuiltinOptions(), n)]  = n

def quantization2dict(quantization):
    if quantization:
        quantization_dict = {}

        def add_array_to_dict(key, data):
            # if data is an int then it means the key is missing
            if not isinstance(data, int):
                quantization_dict[key] = data.tolist()

        add_array_to_dict('min', quantization.MinAsNumpy())
        add_array_to_dict('max', quantization.MaxAsNumpy())
        add_array_to_dict('scale', quantization.ScaleAsNumpy())
        add_array_to_dict('zero_point', quantization.ZeroPointAsNumpy())
        quantization_dict['details_type'] = quantization.DetailsType()
        quantization_dict['quantized_dimension'] = quantization.QuantizedDimension()

        return quantization_dict
    else:
        return None

def builtinoptions2dict(table, type_):
    def camel2snake(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    if table:
        # lookup the builtin options class type
        class_type = BUILTINOPTIONS_TYPES[type_]
        # import the class
        bio_class = getattr(importlib.import_module(f'.schema_generated.{class_type}', package=__package__), class_type)
        # instantiate the class
        bio_inst = bio_class()
        bio_inst.Init(table.Bytes, table.Pos)
        skipped_attrs = set([
            'Init',
            '_tab',
             f'GetRootAs{class_type}',
            f'{class_type}BufferHasIdentifier'
        ]) # all builtin option classes have these attrs so skip them

        builtinoptions_dict = {}
        for a in [a for a in dir(bio_inst) if not a.startswith('__') and a not in skipped_attrs]:
            builtinoptions_dict[camel2snake(a)] = getattr(bio_inst, a)()

        return builtinoptions_dict
    else:
        return None

def customoptions2dict(data, format):
    #TODO: convert to dict, for not just return the array
    if not isinstance(data, int):
        return data.tolist()

    return None

def read_flatbuffer(model_filename):
    if not os.path.exists(model_filename):
        raise FileNotFoundError(
            'File cannot be found at {model_filename}')
    
    fb_model = None

    with open(model_filename, 'rb') as fd:
        buf = fd.read()
        fb_model = Model.Model.GetRootAsModel(buf, 0)

    if not fb_model:
        raise RuntimeError(
            'Unable to load {model_filename}')

    metadata = []
    for metadata_index in range(fb_model.MetadataLength()):
        m = fb_model.Metadata(metadata_index)
        metadata.append({'name': m.Name(), 'buffer': m.Buffer()})

    xc_model = xcore_model.XCOREModel(
        version = fb_model.Version(),
        description = fb_model.Description(),
        metadata = metadata
    )

    # load operator_codes
    fb_operator_codes = []
    for i_operator_code in range(fb_model.OperatorCodesLength()):
        fb_operator_codes.append(fb_model.OperatorCodes(i_operator_code))

    # load buffers
    for i_buffer in range(fb_model.BuffersLength()):
        fb_buffer = fb_model.Buffers(i_buffer)
        if fb_buffer.DataLength() == 0:
            xc_model.create_buffer()
        else:
            xc_model.create_buffer(fb_buffer.DataAsNumpy().tolist())

    # load subgraphs
    for i_subgraph in range(fb_model.SubgraphsLength()):
        fb_subgraph = fb_model.Subgraphs(i_subgraph)
        xc_subgraph = xc_model.create_subgraph()

        fb_subgraph_inputs = []
        for i_input in range(fb_subgraph.InputsLength()):
            fb_subgraph_inputs.append(fb_subgraph.Inputs(i_input))

        fb_subgraph_outputs = []
        for i_output in range(fb_subgraph.OutputsLength()):
            fb_subgraph_outputs.append(fb_subgraph.Outputs(i_output))

        # load tensors
        for i_tensor in range(fb_subgraph.TensorsLength()):
            fb_tensor = fb_subgraph.Tensors(i_tensor)

            # create the tensor
            xc_subgraph.create_tensor(
                fb_tensor.Name().decode('utf-8'),
                xcore_model.TensorType(fb_tensor.Type()),
                fb_tensor.ShapeAsNumpy().tolist(),
                buffer = xc_model.buffers[fb_tensor.Buffer()],
                quantization=quantization2dict(fb_tensor.Quantization()),
                isinput = i_tensor in fb_subgraph_inputs,
                isoutput = i_tensor in fb_subgraph_outputs,
            )

        # load operators
        for i_operator in range(fb_subgraph.OperatorsLength()):
            fb_operator = fb_subgraph.Operators(i_operator)

            fb_operator_code = fb_operator_codes[fb_operator.OpcodeIndex()]

            if fb_operator_code.CustomCode():
                custom_opcode = OperatorCodes.XCOREOpCodes(fb_operator_code.CustomCode().decode('utf-8'))
            else:
                custom_opcode = None

            xc_operator_code = OperatorCodes.OperatorCode(
                OperatorCodes.BuiltinOpCodes(fb_operator_code.BuiltinCode()),
                custom_opcode=custom_opcode,
                version=fb_operator_code.Version()
            )

            xc_operator_inputs = []
            for i_input in range(fb_operator.InputsLength()):
                i_tensor = fb_operator.Inputs(i_input)
                xc_operator_inputs.append(xc_subgraph.tensors[i_tensor])

            xc_operator_outputs = []
            for i_output in range(fb_operator.OutputsLength()):
                i_tensor = fb_operator.Outputs(i_input)
                xc_operator_outputs.append(xc_subgraph.tensors[i_tensor])

            xc_builtin_options = builtinoptions2dict(fb_operator.BuiltinOptions(), fb_operator.BuiltinOptionsType())
            xc_custom_options = customoptions2dict(fb_operator.CustomOptionsAsNumpy(), fb_operator.CustomOptionsFormat())

            xc_subgraph.create_operator(
                xc_operator_code,
                inputs = xc_operator_inputs,
                outputs = xc_operator_outputs,
                builtin_options = xc_builtin_options,
                custom_options = xc_custom_options
            )

    return xc_model