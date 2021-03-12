# Copyright 2019-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pathlib
import flatbuffers
import logging
from typing import (
    Dict,
    Any,
    Union,
    Optional,
    Iterable,
    List,
    Counter,
    TypeVar,
    Type,
    overload,
    cast,
    MutableSequence,
)

from . import (
    _IRObject,
    _ModelDependent,
    _DataContainer,
    OperatorCode,
    Buffer,
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

from tflite2xcore.execution_planning import ReverseDepthFirstPlanner

_R = TypeVar("_R", bound="XCOREModel")

_T = TypeVar("_T", bound="_ModelDependent")


class _ModelDependentContainer(MutableSequence[_T]):
    def __init__(self, model: "XCOREModel", objects: Optional[Iterable[_T]] = None):
        self._model = model
        self._objects: List[_T] = []
        if objects:
            self.extend(objects)  # pylint: disable=no-member

    @overload
    def __getitem__(self, key: int) -> _T:
        ...

    @overload
    def __getitem__(self, key: slice) -> "_ModelDependentContainer[_T]":
        ...

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union[_T, "_ModelDependentContainer[_T]"]:
        if isinstance(key, int):
            return self._objects[key]

        return _ModelDependentContainer(self._model, self._objects[key])

    def __delitem__(self, key: Union[int, slice]) -> None:
        objects = [self[key]] if isinstance(key, int) else self[key]
        for obj in objects:
            del obj._model
        del self._objects[key]

    @overload
    def __setitem__(self, key: int, obj: _T) -> None:
        ...

    @overload
    def __setitem__(self, key: slice, objects: Iterable[_T]) -> None:
        ...

    def __setitem__(
        self, key: Union[int, slice], objects: Union[_T, Iterable[_T]]
    ) -> None:
        # NOTE: mypy cannot correctly infer the type of objects given the type of key
        # so the casts below are to suppress the resulting mypy errors

        if isinstance(key, int):
            del self._objects[key]._model
            obj = cast(_T, objects)
            self._objects[key] = obj
            obj._model = self._model
            return

        # NOTE: since key must be a slice now, there is guarantee that
        # self._objects[key] has the same length as objects

        old_objects = self._objects[key]
        for old_obj in old_objects:
            del old_obj._model

        # objects can be an iterator, so we need to set _model on the fly
        def set_model(obj: _T) -> _T:
            obj._model = self._model
            return obj

        self._objects[key] = (set_model(obj) for obj in cast(Iterable[_T], objects))

    def insert(self, idx: int, obj: _T) -> None:
        self._objects.insert(idx, obj)
        obj._model = self._model

    def __len__(self) -> int:
        return len(self._objects)


class XCOREModel(_IRObject):
    def __init__(
        self,
        version: Optional[int] = None,
        description: Optional[str] = None,
        subgraphs: Optional[Iterable[Subgraph]] = None,
        buffers: Optional[Iterable[Buffer]] = None,
        metadata: Optional[Iterable[Metadata]] = None,
    ) -> None:
        super().__init__()
        self.version = version or 3
        self.description = description or ""
        self.buffers = _ModelDependentContainer[Buffer](self, buffers)
        self.metadata = _ModelDependentContainer[Metadata](self, metadata)
        self.subgraphs = _ModelDependentContainer[Subgraph](self, subgraphs)

    def register_dependent(self, dependent: _ModelDependent) -> None:
        if isinstance(dependent, Buffer):
            self.buffers.append(dependent)  # pylint: disable=no-member
        elif isinstance(dependent, Metadata):
            self.metadata.append(dependent)  # pylint: disable=no-member
        elif isinstance(dependent, Subgraph):
            self.subgraphs.append(dependent)  # pylint: disable=no-member
        else:
            raise TypeError(f"Unsupported model dependent with type {type(dependent)}")

    def is_equal(self, other: Any) -> bool:
        return (
            super().is_equal(other)
            and self.version == other.version
            # and self.description == other.description  # intentionally not compared
            and self.sequence_equal(self.buffers, other.buffers)
            and self.sequence_equal(self.subgraphs, other.subgraphs)
            and self.sequence_equal(self.metadata, other.metadata)
        )

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

        # load metadata
        metadata_map = {
            metadataT.buffer: Metadata(
                name=metadataT.name.decode("utf-8")  # type: ignore
                if metadataT.name
                else None,
                model=model,
                data=modelT.buffers[metadataT.buffer].data,
            )
            for metadataT in modelT.metadata or []
        }

        # check that buffer 0 is empty
        if len(modelT.buffers[0].data) > 0:  # NOTE: BufferT.data can be np.ndarray
            logging.warning("Non-empty buffer 0 in flatbuffer!")

        # create all non-metadata buffers
        buffer_map = {
            idx: Buffer(model, bufferT.data)
            for idx, bufferT in enumerate(modelT.buffers)
            if idx not in metadata_map
        }

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
            subgraph = Subgraph(
                name=subgraphT.name.decode("utf-8")  # type: ignore
                if subgraphT.name
                else None,
                model=model,
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

                if tensorT.buffer in metadata_map:
                    # a tensor is referencing a metadata buffer
                    # this shouldn't happen, but we can work around it
                    metadata = metadata_map[tensorT.buffer]
                    logging.warning(
                        f"Tensor {tensor_index} referencing "
                        f'metadata "{metadata.name}" with buffer {tensorT.buffer}'
                    )
                    buffer_map[tensorT.buffer] = Buffer(model, metadata.data)

                tensor = subgraph.create_tensor(
                    name=tensorT.name.decode("utf-8"),  # type: ignore
                    type_=TensorType(tensorT.type),
                    shape=tensorT.shape,
                    buffer=buffer_map[tensorT.buffer],
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

        modelT.buffers = []

        def create_buffer_from_container(data_container: _DataContainer) -> int:
            """ returns the index of the serialized bufferT object"""
            bufferT = schema.BufferT()  # type: ignore
            if len(data_container.data) > 0:
                bufferT.data = data_container.data
            modelT.buffers.append(bufferT)
            return len(modelT.buffers) - 1

        # check that buffer 0 is empty
        if self.buffers[0]:
            logging.warning("Non-empty buffer 0 in model!")

        # create tensor buffers
        buffer_idx_map: Dict[Buffer, int] = {}
        for buffer in self.buffers:
            buffer_idx_map[buffer] = create_buffer_from_container(buffer)

        # create metadata and their buffers
        modelT.metadata = []
        for metadata in self.metadata:
            metadataT = schema.MetadataT()  # type: ignore
            metadataT.name = metadata.name
            metadataT.buffer = create_buffer_from_container(metadata)
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
                tensorT.buffer = buffer_idx_map[tensor.buffer]
                tensorT.type = tensor.type.value
                if tensor.quantization:
                    tensorT.quantization = dict_to_quantization(tensor.quantization)
                subgraphT.tensors.append(tensorT)

            # set operators
            subgraphT.operators = []
            planner = ReverseDepthFirstPlanner(subgraph)
            for operator in planner.make_plan():
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
