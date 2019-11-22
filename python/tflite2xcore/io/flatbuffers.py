# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import sys

from .schema_generated import Model
from .. import xcore_model
from .. import OperatorCodes

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
            xc_model.create_buffer(list(fb_buffer.DataAsNumpy()))

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
            xc_subgraph.create_tensor(
                fb_tensor.Name().decode('utf-8'),
                xcore_model.TensorType(fb_tensor.Type()),
                list(fb_tensor.ShapeAsNumpy()),
                buffer = xc_model.buffers[fb_tensor.Buffer()],
                quantization=None, #TODO: quantization
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

            xc_subgraph.create_operator(
                xc_operator_code,
                inputs = xc_operator_inputs,
                outputs = xc_operator_outputs,
                builtin_options=None, #TODO: load builtin_options
                custom_options=None #TODO: load custom_options
            )

    return xc_model