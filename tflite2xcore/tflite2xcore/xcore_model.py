# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import logging
import numpy as np  # type: ignore
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Union, Optional, Iterable, Dict, Tuple, List

from tflite2xcore.xcore_schema import TensorType, OperatorCode
from tflite2xcore.serialization.flatbuffers_io import XCORESerializationMixin


_BufferDataType = Union[None, list, tuple, bytes, bytearray, np.ndarray]
_IntType = Union[int, np.integer]
_ShapeType = Union[None, Iterable[_IntType], np.ndarray]
OptionsType = Dict[str, Any]


class _AbstractContainer(ABC):
    @abstractmethod
    def sanity_check(self) -> None:
        raise NotImplementedError()

    def is_equal(self, other: Any) -> bool:
        self.sanity_check()
        return type(self) is type(other)


class Buffer(_AbstractContainer):
    def __init__(
        self,
        model: "XCOREModel",
        data: _BufferDataType = None,
        *,
        owners: Optional[Iterable["Tensor"]] = None,
    ) -> None:
        # Generally, do not use this constructor to instantiate Buffer!
        # Use XCOREModel.create_buffer instead.

        self.model = model  # parent
        self.data = data  # type: ignore # see https://github.com/python/mypy/issues/3004
        self.owners: List["Tensor"] = list(owners or [])

    @property
    def data(self) -> bytes:
        return self._data

    @data.setter
    def data(self, data: _BufferDataType) -> None:
        if data is None:
            self._data = b""
        elif isinstance(data, (list, tuple, bytes, bytearray)):
            # this ensures immutability and that lists have uint8 elements only
            self._data = bytes(data)
        elif isinstance(data, np.ndarray):
            try:
                TensorType.from_numpy_dtype(data.dtype)
            except KeyError:
                # we throw a warning if a non-convertible datatype is used
                logging.getLogger("XCOREModel").warning(
                    f"Numpy array of type {data.dtype} stored in buffer"
                )
            self._data = data.tobytes()
        else:
            raise TypeError(f"data must be list/tuple of bytes or numpy array")

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return f"Buffer[{len(self.data)}]"

    def is_equal(self, other: Any) -> bool:
        return (
            super().is_equal(other)
            and len(self.owners) == len(other.owners)  # avoids circular dependencies
            and self.data == other.data
        )

    def sanity_check(self) -> None:
        assert self in self.model.buffers
        for owner in self.owners:
            assert owner.buffer is self


class Operator(_AbstractContainer):
    def __init__(
        self,
        subgraph: "Subgraph",
        operator_code: OperatorCode,
        name: Optional[str] = None,
        inputs: Optional[Iterable["Tensor"]] = None,
        outputs: Optional[Iterable["Tensor"]] = None,
        builtin_options: Optional[OptionsType] = None,
        custom_options: Optional[OptionsType] = None,
    ):
        # Generally, do not use this constructor to instantiate Operator!
        # Use Subgraph.create_operator instead.

        self.subgraph = subgraph  # parent
        self.operator_code = operator_code
        self.name = name
        self.inputs = list(inputs or [])
        self.outputs = list(outputs or [])
        self.builtin_options = builtin_options or {}
        self.custom_options = custom_options or {}

    def add_custom_options(self, **kwargs: Any) -> None:
        if kwargs:
            self.custom_options.update(kwargs)

    @property
    def model(self) -> "XCOREModel":
        return self.subgraph.model

    def __str__(self) -> str:
        return f"({self.subgraph.operators.index(self)}) operator_code={self.operator_code}"

    def is_equal(self, other: Any) -> bool:
        return (
            super().is_equal(other)
            and self.operator_code == other.operator_code
            and self.name == other.name
            and self.inputs == other.inputs
            and self.outputs == other.outputs
            and self.builtin_options == other.builtin_options
            and self.custom_options == other.custom_options
        )

    def sanity_check(self) -> None:
        assert self in self.subgraph.operators
        # check for duplicates
        assert len(self.inputs) == len(set(self.inputs))
        assert len(self.outputs) == len(set(self.outputs))
        # check double links with inputs/outputs
        for tensor in self.inputs:
            assert self in tensor.consumers
        for tensor in self.outputs:
            assert self in tensor.producers


class Tensor(_AbstractContainer):
    buffer: Buffer

    def __init__(
        self,
        subgraph: "Subgraph",
        name: str,
        type_: TensorType,
        shape,
        buffer: Optional[Buffer] = None,
        quantization: Optional[OptionsType] = None,
        producers: Optional[Iterable[Operator]] = None,
        consumers: Optional[Iterable[Operator]] = None,
    ):
        # Generally, do not use this constructor to instantiate Tensor!
        # Use Subgraph.create_tensor instead.
        self.subgraph = subgraph  # parent
        self.name = name
        assert isinstance(type_, TensorType)
        self.type = type_
        self.shape = shape

        if buffer is None:
            self.buffer = self.model.create_buffer()
        else:
            assert isinstance(buffer, Buffer)
            assert buffer in self.model.buffers
            self.buffer = buffer
        self.buffer.owners.append(self)

        self.quantization = quantization or {}
        self.producers: List[Operator] = list(producers or [])
        self.consumers: List[Operator] = list(consumers or [])

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @shape.setter
    def shape(self, shape: _ShapeType) -> None:
        if shape is None:
            shape = []
        elif isinstance(shape, np.ndarray):
            shape = shape.tolist()
        else:
            shape = list(shape)

        for j, s in enumerate(shape):
            if not isinstance(s, (int, np.integer)):
                raise TypeError(
                    "Tensor.shape must be an iterable of integers, "
                    f"got shape[{j}] = {s} with type {type(s)}"
                )

        self._shape = tuple(int(s) for s in shape)

    @property
    def model(self) -> "XCOREModel":
        return self.subgraph.model

    def is_equal(self, other: Any) -> bool:
        return (
            super().is_equal(other)
            and self.name == other.name
            and self.type == other.type
            and self.shape == other.shape
            and self.buffer == other.buffer
            and self.quantization == other.quantization
            and len(self.producers) == len(other.producers)  # avoids circular deps
            and len(self.consumers) == len(other.consumers)  # avoids circular deps
        )

    def __str__(self) -> str:
        return f"name={self.name}, type={self.type.name}, shape={self.shape}, buffer={self.buffer}"

    def sanity_check(self) -> None:
        assert self in self.subgraph.tensors
        assert self in self.buffer.owners
        # check for duplicates
        assert len(self.consumers) == len(set(self.consumers))
        assert len(self.producers) == len(set(self.producers))
        # check double links with consumers/producers
        for op in self.producers:
            assert self in op.outputs
        for op in self.consumers:
            assert self in op.inputs

    @property
    def sanitized_name(self) -> str:
        """Return a name that is safe to use in source code"""
        return self.name.replace("/", "_")

    @property
    def size(self) -> int:
        return self.type.to_bytes() * np.prod(self.shape)

    def as_array(self, dtype: Optional[type] = None) -> np.ndarray:
        arr = np.frombuffer(self.buffer._data, dtype=self.type.to_numpy_dtype())
        if dtype:
            arr = arr.astype(dtype)
        return arr.reshape(self.shape)

    @property
    def is_constant(self) -> bool:
        # There is an esoteric case where by a tensor without any producers could potentially be
        # modified if it shares a buffer with a tensor from another subgraph.
        # As such we also check if all owners of its buffer have no producers and are not inputs
        return all(
            not t.producers and t not in self.subgraph.inputs
            for t in self.buffer.owners
        )


class Subgraph:
    def __init__(
        self, model, name=None, inputs=None, outputs=None, operators=None, tensors=None
    ):
        # Generally, do not use this constructor to instantiate Subgraph!
        # Use XCOREModel.create_subgraph instead.
        self.model = model  # parent
        self.name = name
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.operators = operators or []
        self.tensors = tensors or []

    @property
    def intermediates(self):
        # intermediates are any tensors that are not an input or an output
        return [t for t in self.tensors if t not in (self.inputs + self.outputs)]

    def create_tensor(
        self,
        name,
        type_,
        shape,
        *,
        buffer=None,
        quantization=None,
        isinput=False,
        isoutput=False,
        producers=None,
        consumers=None,
    ):

        name = self.make_unique_tensor_name(name)
        tensor = Tensor(
            self, name, type_, shape, buffer, quantization, producers, consumers
        )
        self.tensors.append(tensor)
        if isinput:
            self.inputs.append(tensor)
        if isoutput:
            self.outputs.append(tensor)
        return tensor

    def remove_tensor(self, tensor):
        assert tensor in self.tensors
        self.tensors.remove(tensor)
        if tensor in self.inputs:
            self.inputs.remove(tensor)
        if tensor in self.outputs:
            self.outputs.remove(tensor)
        for op in tensor.consumers:
            try:
                op.inputs.remove(tensor)
            except ValueError:
                pass
        for op in tensor.producers:
            try:
                op.outputs.remove(tensor)
            except ValueError:
                pass
        tensor.consumers, tensor.producers = [], []
        tensor.buffer.owners.remove(tensor)
        tensor.subgraph = tensor.buffer = None

    def generate_unique_op_name(self, operator_code):
        existing_names = [op.name for op in self.operators]
        j = 0
        while True:
            j, new_name = j + 1, f"{operator_code.name}_{j}"
            if new_name not in existing_names:
                return new_name

    def make_unique_tensor_name(self, candidate_name):
        existing_names = [
            name
            for tensor in self.tensors
            for name in (tensor.name, tensor.sanitized_name)
        ]

        j, new_name = 1, candidate_name
        while True:
            if new_name not in existing_names:
                return new_name
            j, new_name = j + 1, f"{candidate_name}_{j}"

    def create_operator(
        self,
        operator_code,
        *,
        inputs=None,
        outputs=None,
        builtin_options=None,
        custom_options=None,
    ):
        name = self.generate_unique_op_name(operator_code)
        operator = Operator(
            self, operator_code, name, inputs, outputs, builtin_options, custom_options,
        )
        self.operators.append(operator)
        for input_tensor in operator.inputs:
            input_tensor.consumers.append(operator)
        for output_tensor in operator.outputs:
            output_tensor.producers.append(operator)
        return operator

    def remove_operator(self, op):
        assert op in self.operators
        self.operators.remove(op)
        for t in op.inputs:
            try:
                t.consumers.remove(op)
            except ValueError:
                pass
        for t in op.outputs:
            try:
                t.producers.remove(op)
            except ValueError:
                pass
        op.inputs, op.outputs = [], []
        op.subgraph = None

    def insert_operator(self, ref_op, new_op, after=False):
        """NOTE: this does not rewire inputs/outputs"""
        # find location of reference op
        try:
            ref_idx = self.operators.index(ref_op)
        except ValueError as e:
            raise ValueError("Cannot find reference operator in the subgraph") from e

        # remove new_op from list if already in the subgraph
        if new_op in self.operators:
            self.operators.remove(new_op)

        # (re)insert new op before/after reference op
        self.operators.insert(ref_idx + (1 if after else 0), new_op)

    def replace_operator(self, op, new_op):
        """NOTE: this does not rewire inputs/outputs"""
        # insert new op
        try:
            self.insert_operator(op, new_op)
        except ValueError:
            raise ValueError("Cannot find operator to replace in the subgraph")
        # remove old op
        self.remove_operator(op)

    def get_tensor(self, name):
        for t in self.tensors:
            if t.name == name:
                return t
        raise ValueError(f"Tensor with name {name} not found!")

    def sanity_check(self):
        assert self in self.model.subgraphs
        # check for duplicates
        assert len(self.inputs) == len(set(self.inputs))
        assert len(self.outputs) == len(set(self.outputs))
        assert len(self.operators) == len(set(self.operators))
        assert len(self.tensors) == len(set(self.tensors))
        # make sure inputs and outputs are not misplaced
        for tensor in self.inputs + self.outputs:
            assert tensor in self.tensors
        # the subgraph is sane as long as all its objects are sane
        for op in self.operators:
            op.sanity_check()
        for tensor in self.tensors:
            tensor.sanity_check()


class Metadata:
    def __init__(self, model, name, buffer=None):
        # Generally, do not use this constructor to instantiate Metadata!
        # Use XCOREModel.create_metadata instead.
        self.model = model  # parent
        self.name = name
        if buffer:
            assert isinstance(buffer, Buffer)
            assert buffer in self.model.buffers
            self.buffer = buffer
        else:
            self.buffer = self.model.create_buffer()
        self.buffer.owners.append(self)

    def __str__(self):
        return f"name={self.name}, buffer={self.buffer}"

    def sanity_check(self):
        assert self in self.buffer.owners


class XCOREModel(XCORESerializationMixin):
    def __init__(
        self,
        version=None,
        description=None,
        subgraphs=None,
        buffers=None,
        metadata=None,
    ):
        self.version = version or 3
        self.description = description or ""
        self.buffers = buffers or []
        self.subgraphs = subgraphs or []
        self.metadata = metadata or []

    def create_buffer(self, data=None) -> Buffer:
        buffer = Buffer(self, data)
        self.buffers.append(buffer)
        return buffer

    def create_metadata(self, name, buffer=None):
        metadata = Metadata(self, name, buffer)
        self.metadata.append(metadata)
        return metadata

    def create_subgraph(self, name=None):
        subgraph = Subgraph(self, name)
        self.subgraphs.append(subgraph)
        return subgraph

    @property
    def operator_codes(self):
        # sort the operators codes from most frequent to least frequent
        #   why? because the flatbuffer is a tiny bit smaller if we do
        counter = Counter()

        for subgraph in self.subgraphs:
            for operator in subgraph.operators:
                counter[operator.operator_code] += 1

        sorted_operator_codes = [op_code for op_code, _ in counter.most_common()]

        return sorted_operator_codes

    @property
    def data_size(self):
        nbytes = 0
        for buffer in self.buffers:
            nbytes += len(buffer)
        return nbytes

    def sanity_check(self):
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
