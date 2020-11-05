# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

import pathlib
import flatbuffers
from typing import Dict, Any, Union, Optional, Iterable, List, Counter, TypeVar, Type

from . import (
    _IRObject,
    OperatorCode,
    Buffer,
    _BufferOwnerContainer,
    _BufferDataType,
    Subgraph,
    Metadata,
    BuiltinOpCodes,
    XCOREOpCodes,
    OperatorCode,
    ExternalOpCodes,
    ValidOpCodes,
    TensorType,
    schema_py_generated as schema,
)
from .dict_conversion import (
    builtin_options_to_dict,
    dict_to_builtin_options,
    quantization_to_dict,
    dict_to_quantization,
    create_dict_from_model,
)
from .flexbuffers import FlexbufferBuilder, FlexbufferParser
from .builtin_options import BuiltinOptions

_R = TypeVar("_R", bound="XCOREModel")


class XCOREModel(_IRObject):
    def __init__(
        self,
        version: Optional[int] = None,
        description: Optional[str] = None,
        subgraphs: Optional[Iterable[Subgraph]] = None,
        buffers: Optional[Iterable[Buffer[_BufferOwnerContainer]]] = None,
        metadata: Optional[Iterable[Metadata]] = None,
    ) -> None:
        super().__init__()
        self.version = version or 3
        self.description = description or ""
        self.buffers = list(buffers or [])
        self.subgraphs = list(subgraphs or [])
        self.metadata = list(metadata or [])

    def is_equal(self, other: Any) -> bool:
        return (
            super().is_equal(other)
            and self.version == other.version
            # and self.description == other.description  # intentionally not compared
            and self.sequence_equal(self.buffers, other.buffers)
            and self.sequence_equal(self.subgraphs, other.subgraphs)
            and self.sequence_equal(self.metadata, other.metadata)
        )

    def create_buffer(
        self, data: Optional[_BufferDataType] = None
    ) -> Buffer[_BufferOwnerContainer]:
        buffer = Buffer[_BufferOwnerContainer](self, data)
        self.buffers.append(buffer)
        return buffer

    def create_metadata(
        self, name: str, buffer: Optional[Buffer[Metadata]] = None
    ) -> Metadata:
        metadata = Metadata(self, name)
        metadata.buffer = Metadata.create_buffer(self) if buffer is None else buffer
        metadata.buffer.owners.append(metadata)

        self.metadata.append(metadata)
        return metadata

    def create_subgraph(self, name: str = "") -> Subgraph:
        subgraph = Subgraph(self, name)
        self.subgraphs.append(subgraph)
        return subgraph

    def count_operator_codes(self) -> Counter[OperatorCode]:
        return Counter(
            operator.operator_code
            for subgraph in self.subgraphs
            for operator in subgraph.operators
        )

    @property
    def operator_codes(self) -> List[OperatorCode]:
        # sort the operators codes from most frequent to least frequent
        #   why? because the flatbuffer is a tiny bit smaller if we do
        return [op_code for op_code, _ in self.count_operator_codes().most_common()]

    @property
    def data_size(self) -> int:
        return sum(len(buffer) for buffer in self.buffers)

    def sanity_check(self) -> None:
        # check for duplicates
        assert len(self.subgraphs) == len(set(self.subgraphs))
        assert len(self.buffers) == len(set(self.buffers))
        assert len(self.metadata) == len(set(self.metadata))
        # the model is sane as long as all its subgraphs are sane
        for subgraph in self.subgraphs:
            subgraph.sanity_check()
        for buffer in self.buffers:
            buffer.sanity_check()
        for metadata in self.metadata:
            metadata.sanity_check()

    @classmethod
    def _from_flatbuffer_model(cls: Type[_R], modelT: schema.ModelT) -> _R:
        model = cls(
            version=modelT.version,
            description=modelT.description.decode("utf-8")  # type: ignore
            if modelT.description
            else None,
        )

        # create buffers
        buffers = [model.create_buffer(**vars(bufferT)) for bufferT in modelT.buffers]

        # load metadata
        if modelT.metadata:
            for metadataT in modelT.metadata:
                model.create_metadata(
                    name=metadataT.name.decode("utf-8")  # type: ignore
                    if metadataT.name
                    else None,
                    buffer=buffers[metadataT.buffer],
                )

        # create operator codes lookup
        operator_codes_lut = []
        for operator_codeT in modelT.operatorCodes:
            opcode: ValidOpCodes = BuiltinOpCodes(operator_codeT.builtinCode)
            if opcode is BuiltinOpCodes.CUSTOM:
                custom_code = operator_codeT.customCode.decode("utf-8")  # type: ignore
                try:
                    opcode = XCOREOpCodes(custom_code)
                except ValueError:
                    opcode = ExternalOpCodes.add_new_opcode(custom_code)
            operator_codes_lut.append(
                OperatorCode(opcode, version=operator_codeT.version)
            )

        # load subgraphs
        for subgraph_index, subgraphT in enumerate(modelT.subgraphs):
            subgraph = model.create_subgraph(
                name=subgraphT.name.decode("utf-8")  # type: ignore
                if subgraphT.name
                else None
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
                    name=tensorT.name.decode("utf-8"),  # type: ignore
                    type_=TensorType(tensorT.type),
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
    def deserialize(cls: Type[_R], bits: bytes) -> _R:
        model_obj = schema.Model.GetRootAsModel(bits, 0)  # type: ignore
        modelT = schema.ModelT.InitFromObj(model_obj)  # type: ignore
        return cls._from_flatbuffer_model(modelT)

    @classmethod
    def read_flatbuffer(cls: Type[_R], filename: Union[pathlib.Path, str]) -> _R:
        with open(pathlib.Path(filename).resolve(), "rb") as fd:
            bits = bytes(fd.read())

        return cls.deserialize(bits)

    def _to_flatbuffer_model(self) -> schema.ModelT:
        modelT = schema.ModelT()  # type: ignore
        modelT.version = self.version
        modelT.description = self.description

        # create buffers
        modelT.buffers = []
        for buffer in self.buffers:
            bufferT = schema.BufferT()  # type: ignore
            if len(buffer.data) > 0:
                bufferT.data = buffer.data
            modelT.buffers.append(bufferT)

        # create metadata
        modelT.metadata = []
        for metadata in self.metadata:
            metadataT = schema.MetadataT()  # type: ignore
            metadataT.name = metadata.name
            metadataT.buffer = self.buffers.index(metadata.buffer)
            modelT.metadata.append(metadataT)

        # create operator_codes
        modelT.operatorCodes = []
        for operator_code in self.operator_codes:
            operatorCodeT = schema.OperatorCodeT()  # type: ignore
            if operator_code.code in BuiltinOpCodes:
                operatorCodeT.builtinCode = operator_code.value
            else:
                operatorCodeT.builtinCode = BuiltinOpCodes.CUSTOM.value
                operatorCodeT.customCode = operator_code.name
            operatorCodeT.version = operator_code.version
            modelT.operatorCodes.append(operatorCodeT)

        # create subgraphs
        modelT.subgraphs = []
        for subgraph in self.subgraphs:
            subgraphT = schema.SubGraphT()  # type: ignore
            subgraphT.name = subgraph.name

            # set inputs and outputs
            subgraphT.inputs = [subgraph.tensors.index(t) for t in subgraph.inputs]
            subgraphT.outputs = [subgraph.tensors.index(t) for t in subgraph.outputs]

            # set tensors
            subgraphT.tensors = []
            for tensor in subgraph.tensors:
                tensorT = schema.TensorT()  # type: ignore
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
                operatorT = schema.OperatorT()  # type: ignore
                op_code = operator.operator_code
                operatorT.opcodeIndex = self.operator_codes.index(op_code)

                operatorT.inputs = [subgraph.tensors.index(t) for t in operator.inputs]
                operatorT.outputs = [
                    subgraph.tensors.index(t) for t in operator.outputs
                ]

                # TODO: fix this hack
                # we need a better data structure to represent inputs/outputs of operators
                if op_code.code is ExternalOpCodes.LceBconv2d:
                    if len(operatorT.inputs) == 3:
                        # bitpacked output
                        operatorT.inputs = (
                            operatorT.inputs[:2] + [-1, -1] + operatorT.inputs[-1:]
                        )
                    else:
                        # int8 output
                        operatorT.inputs = operatorT.inputs + [-1]

                if op_code.code in BuiltinOpCodes:
                    builtin_options_type = BuiltinOptions.from_BuiltinOpCodes(
                        op_code.code
                    )
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

        return modelT  # type: ignore

    def serialize(self) -> bytes:
        modelT = self._to_flatbuffer_model()
        builder = flatbuffers.Builder(1024 * 1024)
        model_offset = modelT.Pack(builder)  # type: ignore
        builder.Finish(model_offset, file_identifier=b"TFL3")
        return bytes(builder.Output())

    def write_flatbuffer(self, filename: Union[pathlib.Path, str]) -> int:
        with open(pathlib.Path(filename).resolve(), "wb") as fd:
            return fd.write(self.serialize())

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[Any, Any]:
        return create_dict_from_model(self, *args, **kwargs)
