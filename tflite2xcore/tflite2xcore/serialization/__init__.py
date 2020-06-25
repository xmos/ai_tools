# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

from .flatbuffers_io import write_flatbuffer, read_flatbuffer
from .xcore_schema import (
    Padding,
    TensorType,
    OperatorCode,
    BuiltinOpCodes,
    CustomOpCode,
    XCOREOpCodes,
    BuiltinOptions,
)
from .flatbuffers_c import FlexbufferParser
