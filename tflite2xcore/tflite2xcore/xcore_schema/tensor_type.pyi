# Copyright (c) 2020, XMOS Ltd, All rights reserved

import enum
import numpy as np
from typing import Callable, Any


class TensorType(enum.IntEnum):
    STRING: TensorType
    FLOAT64: TensorType
    FLOAT32: TensorType
    FLOAT16: TensorType
    COMPLEX64: TensorType
    INT64: TensorType
    INT32: TensorType
    INT16: TensorType
    INT8: TensorType
    UINT8: TensorType
    BOOL: TensorType

    @classmethod
    def __call__(cls, x: Any) -> TensorType:
        ...

    def to_stdint_type(self) -> str:
        ...

    def sizeof(self) -> int:
        ...

    def to_numpy_dtype(self) -> np.dtype:
        ...

    from_numpy_dtype: Callable[[np.dtype], TensorType]
