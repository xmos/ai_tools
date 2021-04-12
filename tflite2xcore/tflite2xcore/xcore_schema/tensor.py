# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import numpy as np
from typing import TYPE_CHECKING, Optional, Iterable, Union, List, Tuple, Any

from . import (
    TensorType,
    _SubgraphDependent,
    Buffer,
    Operator,
    _OpOptionsType,
)

if TYPE_CHECKING:
    from . import Subgraph

_ShapeInputType = Union[None, Iterable[Union[int, np.integer]], np.ndarray]


class Tensor(_SubgraphDependent):
    name: str
    buffer: Buffer

    def __init__(
        self,
        subgraph: "Subgraph",
        name: str,
        type_: TensorType,
        shape: _ShapeInputType,
        quantization: Optional[_OpOptionsType] = None,
        producers: Optional[Iterable[Operator]] = None,
        consumers: Optional[Iterable[Operator]] = None,
        custom_options: Optional[_OpOptionsType] = None,
    ) -> None:
        # Generally, do not use this constructor to instantiate Tensor!
        # Use Subgraph.create_tensor instead.

        super().__init__(name or "")
        self._subgraph = subgraph  # parent
        assert isinstance(type_, TensorType)
        self.type = type_
        self.shape: Tuple[int, ...] = shape  # type: ignore # see https://github.com/python/mypy/issues/3004

        self.quantization = quantization or {}
        self.producers: List[Operator] = list(producers or [])
        self.consumers: List[Operator] = list(consumers or [])
        self.custom_options = custom_options or {}

    def add_custom_options(self, **kwargs: Any) -> None:
        if kwargs:
            self.custom_options.update(kwargs)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @shape.setter
    def shape(self, shape: _ShapeInputType) -> None:
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

    def is_equal(self, other: Any) -> bool:
        return (
            super().is_equal(other)
            # and self.name == other.name  # intentionally not compared
            and self.buffer.is_equal(other.buffer)
            and self.type == other.type
            and self.shape == other.shape
            and self.quantization == other.quantization
            and len(self.producers) == len(other.producers)  # avoids circular deps
            and len(self.consumers) == len(other.consumers)  # avoids circular deps
        )

    def __str__(self) -> str:
        return f"name={self.name}, type={self.type.name}, shape={self.shape}, buffer={self.buffer}"

    def sanity_check(self) -> None:
        assert self in self.subgraph.tensors
        assert self in self.buffer.owners
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
        return self.type.sizeof() * np.prod(self.shape)  # type: ignore

    def as_array(self, dtype: Optional[type] = None) -> np.ndarray:
        arr = np.copy(
            np.frombuffer(self.buffer._data, dtype=self.type.to_numpy_dtype())
        )
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
