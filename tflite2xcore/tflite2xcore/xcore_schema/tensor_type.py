# Copyright (c) 2020, XMOS Ltd, All rights reserved

# type: ignore

import enum
import numpy as np

from . import schema_py_generated as schema

TensorType = enum.IntEnum(
    "TensorType",
    {k: v for k, v in vars(schema.TensorType).items() if not k.startswith("__")},
)

__TensorType_to_stdint_type = {
    # TensorType.STRING: None,  # intentionally not supported
    TensorType.FLOAT64: "float64_t",
    TensorType.FLOAT32: "float32_t",
    TensorType.FLOAT16: "float16_t",
    TensorType.COMPLEX64: "complex64_t",
    TensorType.INT64: "int64_t",
    TensorType.INT32: "int32_t",
    TensorType.INT16: "int16_t",
    TensorType.INT8: "int8_t",
    TensorType.UINT8: "uint8_t",
    TensorType.BOOL: "uint8_t",
}
TensorType.to_stdint_type = lambda self: __TensorType_to_stdint_type[self]

__TensorType_sizeof = {
    # TensorType.STRING: None,  # intentionally not supported
    TensorType.FLOAT64: 8,
    TensorType.FLOAT32: 4,
    TensorType.FLOAT16: 2,
    TensorType.COMPLEX64: 8,
    TensorType.INT64: 8,
    TensorType.INT32: 4,
    TensorType.INT16: 2,
    TensorType.INT8: 1,
    TensorType.UINT8: 1,
    TensorType.BOOL: 1,
}
TensorType.sizeof = lambda self: __TensorType_sizeof[self]

__TensorType_to_numpy_dtype = {
    # TensorType.STRING: None,  # intentionally not supported
    TensorType.FLOAT64: np.dtype(np.float64),
    TensorType.FLOAT32: np.dtype(np.float32),
    TensorType.FLOAT16: np.dtype(np.float16),
    TensorType.COMPLEX64: np.dtype(np.complex64),
    TensorType.INT64: np.dtype(np.int64),
    TensorType.INT32: np.dtype(np.int32),
    TensorType.INT16: np.dtype(np.int16),
    TensorType.INT8: np.dtype(np.int8),
    TensorType.UINT8: np.dtype(np.uint8),
    TensorType.BOOL: np.dtype(np.bool_),
}
TensorType.to_numpy_dtype = lambda self: __TensorType_to_numpy_dtype[self]

__TensorType_from_numpy_dtype = {
    np.dtype(np.float64): TensorType.FLOAT64,
    np.dtype(np.float32): TensorType.FLOAT32,
    np.dtype(np.float16): TensorType.FLOAT16,
    np.dtype(np.complex64): TensorType.COMPLEX64,
    np.dtype(np.int64): TensorType.INT64,
    np.dtype(np.int32): TensorType.INT32,
    np.dtype(np.int16): TensorType.INT16,
    np.dtype(np.int8): TensorType.INT8,
    np.dtype(np.uint8): TensorType.UINT8,
    np.dtype(np.bool_): TensorType.BOOL,
}
TensorType.from_numpy_dtype = lambda x: __TensorType_from_numpy_dtype[np.dtype(x)]
