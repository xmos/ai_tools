# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import numpy as np
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Type,
    Generic,
    Iterable,
    TypeVar,
    Optional,
    Union,
    List,
    Any,
)

from . import _IRObject, TensorType

if TYPE_CHECKING:
    from .xcore_model import XCOREModel


_BufferDataType = Union[list, tuple, bytes, bytearray, np.ndarray]
_T = TypeVar("_T", bound="_BufferOwnerContainer", covariant=True)


class Buffer(_IRObject, Generic[_T]):
    def __init__(
        self,
        model: "XCOREModel",
        data: Optional[_BufferDataType] = None,
        *,
        owners: Optional[Iterable[_T]] = None,
    ) -> None:
        # Generally, do not use this constructor to instantiate Buffer!
        # Use XCOREModel.create_buffer instead.

        self.model = model  # parent
        self.data = data  # type: ignore # see https://github.com/python/mypy/issues/3004
        self.owners: List[_T] = list(owners or [])

    @property
    def data(self) -> bytes:
        return self._data

    @data.setter
    def data(self, data: Optional[_BufferDataType]) -> None:
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


class _BufferOwnerContainer(_IRObject):
    buffer: Buffer["_BufferOwnerContainer"]

    @property
    @abstractmethod
    def model(self) -> "XCOREModel":
        raise NotImplementedError()

    def is_equal(self, other: Any) -> bool:
        return (
            super().is_equal(other)
            # and self.name == other.name  # intentionally not compared
            and self.buffer.is_equal(other.buffer)
        )

    @classmethod
    def create_buffer(
        cls: Type[_T], model: "XCOREModel", data: Optional[_BufferDataType] = None
    ) -> Buffer[_T]:
        buffer = Buffer[_T](model, data)
        model.buffers.append(buffer)
        return buffer
