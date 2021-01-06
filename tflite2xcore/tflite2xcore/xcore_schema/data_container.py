# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import numpy as np
from typing import TYPE_CHECKING, Iterable, Optional, Union, List, Any

from . import _IRObject, _ModelDependent, TensorType

if TYPE_CHECKING:
    from .xcore_model import XCOREModel
    from . import Tensor


_BufferDataType = Union[list, tuple, bytes, bytearray, np.ndarray]


class _DataContainer(_ModelDependent):
    def __init__(
        self, name: Optional[str] = None, data: Optional[_BufferDataType] = None,
    ) -> None:
        super().__init__(name)
        self.data = data  # type: ignore # see https://github.com/python/mypy/issues/3004

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

    def is_equal(self, other: Any) -> bool:
        return super().is_equal(other) and self.data == other.data


class Buffer(_DataContainer):
    def __init__(
        self,
        model: Optional["XCOREModel"] = None,
        data: Optional[_BufferDataType] = None,
        *,
        owners: Optional[Iterable["Tensor"]] = None,
    ) -> None:
        super().__init__(data=data)
        if model:
            model.buffers.append(self)
        self.owners: List["Tensor"] = list(
            owners or []
        )  # TODO: should this be managed by Tensor?

    def __str__(self) -> str:
        return f"Buffer[{len(self.data)}]"

    def is_equal(self, other: Any) -> bool:
        # check owner length only to avoid circular dependencies
        return super().is_equal(other) and len(self.owners) == len(other.owners)

    def sanity_check(self) -> None:
        assert self in self.model.buffers
        for owner in self.owners:
            assert owner.buffer is self


class Metadata(_DataContainer):
    def __init__(
        self,
        name: str,
        model: Optional["XCOREModel"] = None,
        data: Optional[_BufferDataType] = None,
    ) -> None:
        super().__init__(name, data)
        if model:
            model.metadata.append(self)

    def __str__(self) -> str:
        return f"name={self.name}, data={list(self.data)}"

    def sanity_check(self) -> None:
        super().sanity_check
        assert self in self.model.metadata
