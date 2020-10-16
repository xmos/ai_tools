# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

import pathlib
import flatbuffers
from typing import Union, Any

from . import schema_py_generated as schema
from . import xcore_schema
from .dict_conversion import (
    builtin_options_to_dict,
    dict_to_builtin_options,
    quantization_to_dict,
    dict_to_quantization,
    create_dict_from_model,
)
from .flexbuffers import FlexbufferBuilder, FlexbufferParser


class XCORESerializationMixin:
    @classmethod
    def _from_flatbuffer_model(cls, modelT: schema.ModelT) -> "XCORESerializationMixin":
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
            opcode = xcore_schema.BuiltinOpCodes(operator_codeT.builtinCode)
            if opcode is xcore_schema.BuiltinOpCodes.CUSTOM:
                custom_code = operator_codeT.customCode.decode("utf-8")
                try:
                    opcode = xcore_schema.XCOREOpCodes(custom_code)
                except ValueError:
                    opcode = xcore_schema.ExternalOpCodes.add_new_opcode(custom_code)
            operator_codes_lut.append(
                xcore_schema.OperatorCode(opcode, version=operator_codeT.version)
            )

        # load subgraphs
        for subgraph_index, subgraphT in enumerate(modelT.subgraphs):
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

            # load operators & set inputs/outputs (registers op as tensor consumer/producer)
            for operator_index, operatorT in enumerate(subgraphT.operators):
                options = {}
                if (
                    hasattr(operatorT, "builtinOptions")
                    and operatorT.builtinOptions is not None
                ):
                    options["builtin_options"] = builtin_options_to_dict(
                        operatorT.builtinOptions
                    )
                if (
                    hasattr(operatorT, "customOptions")
                    and operatorT.customOptions is not None
                ):
                    options["custom_options"] = FlexbufferParser().parse(
                        bytes(operatorT.customOptions)
                    )

                def is_valid_tensor_index(
                    idx: int, lower: int = -1, upper: int = len(tensors)
                ) -> bool:
                    if idx < lower or idx >= upper:
                        raise ValueError(
                            f"Invalid input tensor index [{idx}]: "
                            f"subgraph [{subgraph_index}], "
                            f"operator [{operator_index}], "
                            f"bounds: [{lower}, {upper}]"
                        )

                    return idx != -1  # -1 encodes optional for input indices

                subgraph.create_operator(
                    operator_code=operator_codes_lut[operatorT.opcodeIndex],
                    inputs=[
                        tensors[input_index]
                        for input_index in operatorT.inputs
                        if is_valid_tensor_index(input_index)
                    ],
                    outputs=[
                        tensors[output_index]
                        for output_index in operatorT.outputs
                        if is_valid_tensor_index(output_index, lower=0)
                    ],
                    **options,
                )

        model.sanity_check()
        return model

    @classmethod
    def deserialize(cls, bits: bytes) -> "XCORESerializationMixin":
        model_obj = schema.Model.GetRootAsModel(bits, 0)
        modelT = schema.ModelT.InitFromObj(model_obj)
        return cls._from_flatbuffer_model(modelT)

    @classmethod
    def read_flatbuffer(
        cls, filename: Union[pathlib.Path, str]
    ) -> "XCORESerializationMixin":
        with open(pathlib.Path(filename).resolve(), "rb") as fd:
            bits = bytes(fd.read())

        return cls.deserialize(bits)

    def _to_flatbuffer_model(self) -> schema.ModelT:
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
            if operator_code.code in xcore_schema.BuiltinOpCodes:
                operatorCodeT.builtinCode = operator_code.value
            else:
                operatorCodeT.builtinCode = xcore_schema.BuiltinOpCodes.CUSTOM.value
                operatorCodeT.customCode = operator_code.name
            operatorCodeT.version = operator_code.version
            modelT.operatorCodes.append(operatorCodeT)

        # create subgraphs
        modelT.subgraphs = []
        for subgraph in self.subgraphs:
            subgraphT = schema.SubGraphT()
            subgraphT.name = subgraph.name

            # set inputs and outputs
            subgraphT.inputs = [subgraph.tensors.index(t) for t in subgraph.inputs]
            subgraphT.outputs = [subgraph.tensors.index(t) for t in subgraph.outputs]

            # set tensors
            subgraphT.tensors = []
            for tensor in subgraph.tensors:
                tensorT = schema.TensorT()
                tensorT.name = tensor.name
                tensorT.shape = tensor.shape
                tensorT.buffer = self.buffers.index(tensor.buffer)
                tensorT.type = tensor.type.value
                if tensor.quantization:
                    tensorT.quantization = dict_to_quantization(tensor.quantization)
                subgraphT.tensors.append(tensorT)

            # set operators
            subgraphT.operators = []
            for operator in subgraph.operators:
                operatorT = schema.OperatorT()
                op_code = operator.operator_code
                operatorT.opcodeIndex = self.operator_codes.index(op_code)

                operatorT.inputs = [subgraph.tensors.index(t) for t in operator.inputs]
                operatorT.outputs = [
                    subgraph.tensors.index(t) for t in operator.outputs
                ]

                if op_code.code in xcore_schema.BuiltinOpCodes:
                    builtin_options_type = op_code.code.to_BuiltinOptions()
                    operatorT.builtinOptionsType = builtin_options_type.value

                    if operator.builtin_options:
                        operatorT.builtinOptions = dict_to_builtin_options(
                            builtin_options_type, operator.builtin_options
                        )

                if operator.custom_options:
                    fbb = FlexbufferBuilder(operator.custom_options)
                    operatorT.customOptions = fbb.get_bytes()
                subgraphT.operators.append(operatorT)

            modelT.subgraphs.append(subgraphT)

        return modelT

    def serialize(self) -> bytes:
        modelT = self._to_flatbuffer_model()
        builder = flatbuffers.Builder(1024 * 1024)
        model_offset = modelT.Pack(builder)
        builder.Finish(model_offset, file_identifier=b"TFL3")
        return bytes(builder.Output())

    def write_flatbuffer(self, filename: Union[pathlib.Path, str]) -> int:
        with open(pathlib.Path(filename).resolve(), "wb") as fd:
            return fd.write(self.serialize())

    def to_dict(self, *args: Any, **kwargs: Any) -> dict:
        return create_dict_from_model(self, *args, **kwargs)
