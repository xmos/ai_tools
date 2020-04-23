# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import json
import pathlib

import flatbuffers
import numpy as np

from . import schema_py_generated as schema
from . import xcore_schema
from .dict_conversion import (
    builtin_options_to_dict,
    dict_to_builtin_options,
    quantization_to_dict,
    create_dict_from_model,
)
from .flatbuffers_c import FlexbufferBuilder, FlexbufferParser


class XCORESerializationMixin:
    @classmethod
    def _from_flatbuffer_model(cls, modelT):
        model = cls(
            version=modelT.version,
            description=modelT.description.decode("utf-8")
            if modelT.description
            else None,
        )

        # create buffers
        buffers = [model.create_buffer(**vars(bufferT)) for bufferT in modelT.buffers]

        # load metadata
        if modelT.metadata:
            for metadataT in modelT.metadata:
                model.create_metadata(
                    name=metadataT.name.decode("utf-8") if metadataT.name else None,
                    buffer=buffers[metadataT.buffer],
                )

        # create operator codes lookup
        operator_codes_lut = []
        for operator_codeT in modelT.operatorCodes:
            if operator_codeT.builtinCode == xcore_schema.BuiltinOpCodes.CUSTOM.value:
                custom_code = operator_codeT.customCode.decode("utf-8")
                if custom_code in xcore_schema.XCOREOpCodes:
                    opcode = xcore_schema.XCOREOpCodes(custom_code)
                else:
                    opcode = xcore_schema.CustomOpCode(custom_code)
            else:
                opcode = xcore_schema.BuiltinOpCodes(operator_codeT.builtinCode)
            operator_codes_lut.append(
                xcore_schema.OperatorCode(opcode, version=operator_codeT.version)
            )

        # load subgraphs
        for subgraphT in modelT.subgraphs:
            subgraph = model.create_subgraph(
                name=subgraphT.name.decode("utf-8") if subgraphT.name else None
            )

            # load tensors
            tensors = []
            for tensor_index, tensorT in enumerate(subgraphT.tensors):
                is_input = tensor_index in subgraphT.inputs
                is_output = tensor_index in subgraphT.outputs

                # load quantization
                quantization = None
                if hasattr(tensorT, "quantization") and tensorT.quantization:
                    quantization = quantization_to_dict(tensorT.quantization)

                tensor = subgraph.create_tensor(
                    name=tensorT.name.decode("utf-8"),
                    type_=xcore_schema.TensorType(tensorT.type),
                    shape=tensorT.shape,
                    buffer=buffers[tensorT.buffer],
                    quantization=quantization,
                    isinput=is_input,
                    isoutput=is_output,
                )
                tensors.append(tensor)

            # load operators & set tensor producer & consumers
            for operatorT in subgraphT.operators:
                operator_code = operator_codes_lut[operatorT.opcodeIndex]
                options = {}
                if (
                    hasattr(operatorT, "builtinOptions")
                    and operatorT.builtinOptions is not None
                ):
                    options["builtin_options"] = builtin_options_to_dict(
                        operatorT.builtinOptions
                    )
                    options["builtin_options_type"] = operatorT.builtinOptionsType

                if (
                    hasattr(operatorT, "customOptions")
                    and operatorT.customOptions is not None
                ):
                    options["custom_options"] = json.loads(
                        FlexbufferParser().parse(bytes(operatorT.customOptions))
                    )

                subgraph.create_operator(
                    operator_code,
                    inputs=[tensors[input_index] for input_index in operatorT.inputs],
                    outputs=[
                        tensors[output_index] for output_index in operatorT.outputs
                    ],
                    **options
                )

        model.sanity_check()
        return model

    @classmethod
    def deserialize(cls, bits):
        model_obj = schema.Model.GetRootAsModel(bits, 0)
        modelT = schema.ModelT.InitFromObj(model_obj)
        return cls._from_flatbuffer_model(modelT)

    @classmethod
    def read_flatbuffer(cls, filename):
        if isinstance(filename, pathlib.Path):
            filename = str(filename.resolve())

        with open(filename, "rb") as fd:
            bits = bytes(fd.read())

        return cls.deserialize(bits)

    def to_flatbuffer_model(self):
        modelT = schema.ModelT()
        modelT.version = self.version
        modelT.description = self.description

        # create buffers
        modelT.buffers = []
        for buffer in self.buffers:
            bufferT = schema.BufferT()
            if len(buffer.data) > 0:
                bufferT.data = buffer.data
            modelT.buffers.append(bufferT)

        # create metadata
        modelT.metadata = []
        for metadata in self.metadata:
            metadataT = schema.MetadataT()
            metadataT.name = metadata.name
            metadataT.buffer = self.buffers.index(metadata.buffer)
            modelT.metadata.append(metadataT)

        # create operator_codes
        modelT.operatorCodes = []
        for operator_code in self.operator_codes:
            operatorCodeT = schema.OperatorCodeT()
            if operator_code.builtin_code:
                operatorCodeT.builtinCode = operator_code.builtin_code.value
            if operator_code.custom_code:
                operatorCodeT.customCode = operator_code.custom_code.name
            operatorCodeT.version = operator_code.version
            modelT.operatorCodes.append(operatorCodeT)

        # create subgraphs
        modelT.subgraphs = []
        for subgraph in self.subgraphs:
            subgraphT = schema.SubGraphT()
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
                tensorT = schema.TensorT()
                tensorT.name = tensor.name
                tensorT.shape = tensor.shape
                tensorT.buffer = self.buffers.index(tensor.buffer)
                tensorT.type = tensor.type.value
                if tensor.quantization:
                    quantizationT = schema.QuantizationParametersT()
                    if "min" in tensor.quantization:
                        quantizationT.min = tensor.quantization["min"]
                    if "max" in tensor.quantization:
                        quantizationT.max = tensor.quantization["max"]
                    if "zero_point" in tensor.quantization:
                        quantizationT.zeroPoint = tensor.quantization["zero_point"]
                    if "scale" in tensor.quantization:
                        quantizationT.scale = tensor.quantization["scale"]
                    if "details_type" in tensor.quantization:
                        if isinstance(tensor.quantization["details_type"], str):
                            quantizationT.detailsType = xcore_schema.QuantizationDetails[
                                tensor.quantization["details_type"]
                            ].value
                        else:
                            quantizationT.detailsType = tensor.quantization[
                                "details_type"
                            ]
                    if "details" in tensor.quantization:
                        quantizationT.details = tensor.quantization["details"]
                    if "quantized_dimension" in tensor.quantization:
                        quantizationT.quantizedDimension = tensor.quantization[
                            "quantized_dimension"
                        ]
                    tensorT.quantization = quantizationT
                subgraphT.tensors.append(tensorT)

            # set operators
            subgraphT.operators = []
            for operator in subgraph.operators:
                operatorT = schema.OperatorT()
                operatorT.opcodeIndex = self.operator_codes.index(
                    operator.operator_code
                )
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
                    operatorT.builtinOptions = dict_to_builtin_options(
                        operator.builtin_options_type, operator.builtin_options
                    )
                if operator.custom_options:
                    fbb = FlexbufferBuilder(operator.custom_options)
                    operatorT.customOptions = fbb.get_bytes()
                subgraphT.operators.append(operatorT)

            modelT.subgraphs.append(subgraphT)

        return modelT

    def serialize(self):
        modelT = self.to_flatbuffer_model()
        builder = flatbuffers.Builder(1024 * 1024)
        model_offset = modelT.Pack(builder)
        builder.Finish(model_offset, file_identifier=b"TFL3")
        return bytes(builder.Output())

    def write_flatbuffer(self, filename):
        if isinstance(filename, pathlib.Path):
            filename = str(filename.resolve())

        with open(filename, "wb") as fd:
            return fd.write(self.serialize())

        return 0

    def to_dict(self, *args, **kwargs):
        return create_dict_from_model(self, *args, **kwargs)


def write_flatbuffer(model, filename):
    assert isinstance(model, XCORESerializationMixin)
    return model.write_flatbuffer(filename)


def read_flatbuffer(filename):
    from tflite2xcore.xcore_model import XCOREModel

    return XCOREModel.read_flatbuffer(filename)
