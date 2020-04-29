# Copyright (c) 2019, XMOS Ltd, All rights reserved

import enum

import numpy as np

from . import schema_py_generated as schema


#  ----------------------------------------------------------------------------
#                                  TensorType
#  ----------------------------------------------------------------------------


TensorType = enum.IntEnum(
    "TensorType",
    {k: v for k, v in vars(schema.TensorType).items() if not k.startswith("__")},
)

__TensorType_to_stdint_type = {
    TensorType.FLOAT32: "float32_t",
    TensorType.FLOAT16: "float16_t",
    TensorType.INT32: "int32_t",
    TensorType.UINT8: "uint8_t",
    TensorType.INT64: "int64_t",
    TensorType.STRING: None,
    TensorType.BOOL: "uint8_t",
    TensorType.INT16: "int16_t",
    TensorType.COMPLEX64: None,
    TensorType.INT8: "int8_t",
}

TensorType.to_stdint_type = lambda self: __TensorType_to_stdint_type[self]

__TensorType_to_bytes = {
    TensorType.FLOAT32: 4,
    TensorType.FLOAT16: 2,
    TensorType.INT32: 4,
    TensorType.UINT8: 1,
    TensorType.INT64: 8,
    TensorType.STRING: None,
    TensorType.BOOL: 1,
    TensorType.INT16: 2,
    TensorType.COMPLEX64: None,
    TensorType.INT8: 1,
}
TensorType.to_bytes = lambda self: __TensorType_to_bytes[self]

__TensorType_to_numpy_type = {
    TensorType.FLOAT32: np.float64,
    TensorType.FLOAT16: np.float64,
    TensorType.INT32: np.int64,
    TensorType.UINT8: np.int64,
    TensorType.INT64: np.int64,
    # TensorType.STRING: None,  # intentionally not supported
    TensorType.BOOL: np.int64,
    TensorType.INT16: np.int64,
    # TensorType.COMPLEX64: None,  # intentionally not supported
    TensorType.INT8: np.int64,
}
TensorType.to_numpy_type = lambda self: __TensorType_to_numpy_type[self]

__TensorType_to_numpy_dtype = {
    TensorType.FLOAT32: np.float32,
    TensorType.FLOAT16: np.single,
    TensorType.INT32: np.int32,
    TensorType.UINT8: np.uint8,
    TensorType.INT64: np.int64,
    # TensorType.STRING: None,  # intentionally not supported
    TensorType.BOOL: np.bool_,
    TensorType.INT16: np.int16,
    # TensorType.COMPLEX64: None,  # intentionally not supported
    TensorType.INT8: np.int8,
}
TensorType.to_numpy_dtype = lambda self: __TensorType_to_numpy_dtype[self]


#  ----------------------------------------------------------------------------
#                               Operator Codes
#  ----------------------------------------------------------------------------


class ValidOpCodes:
    pass


class EnumOpCodes(ValidOpCodes, enum.Enum):
    pass


BuiltinOpCodes = EnumOpCodes(
    "BuiltinOpCodes",
    {k: v for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("__")},
)


class CustomOpCode(ValidOpCodes):
    def __init__(self, name):
        self.name = name
        self.value = name


class XCOREOpCodes(EnumOpCodes):
    # TODO: consider an IntEnum for this instead of strings
    XC_lookup_8 = "XC_lookup_8"
    XC_argmax_16 = "XC_argmax_16"  # currently not used by any passes
    XC_maxpool2d = "XC_maxpool2d"
    XC_avgpool2d = "XC_avgpool2d"
    XC_avgpool2d_global = "XC_avgpool2d_global"
    XC_fc_deepin_anyout = "XC_fc_deepin_anyout"
    XC_requantize_16_to_8 = "XC_requantize_16_to_8"  # currently only used after FC
    XC_conv2d_shallowin = "XC_conv2d_shallowin"
    XC_conv2d_deep = "XC_conv2d_deep"
    XC_conv2d_1x1 = "XC_conv2d_1x1"
    XC_conv2d_depthwise = "XC_conv2d_depthwise"


class OperatorCode:
    def __init__(self, opcode, *, custom_code=None, version=None):
        assert isinstance(opcode, ValidOpCodes), "Invalid opcode!"
        self.version = version or 1

        if isinstance(opcode, XCOREOpCodes) or isinstance(opcode, CustomOpCode):
            self.builtin_code = BuiltinOpCodes.CUSTOM
            self.custom_code = opcode
        else:
            self.builtin_code = opcode
            if self.builtin_code == BuiltinOpCodes.CUSTOM:
                assert isinstance(
                    custom_code, XCOREOpCodes
                ), "Must provide custom_code if builtin_code is 'CUSTOM'!"
                self.custom_code = custom_code
            else:
                self.custom_code = None

    @property
    def code(self):
        return (
            self.custom_code
            if self.builtin_code == BuiltinOpCodes.CUSTOM
            else self.builtin_code
        )

    @property
    def name(self):
        return self.code.name

    def __eq__(self, obj):
        return (
            isinstance(obj, OperatorCode)
            and obj.code == self.code
            and obj.version == self.version
        )

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return f"{self.name} (version {self.version})"


#  ----------------------------------------------------------------------------
#                               Misc Enums
#  ----------------------------------------------------------------------------


ActivationFunctionType = enum.Enum(
    "ActivationFunctionType",
    {
        k: v
        for k, v in vars(schema.ActivationFunctionType).items()
        if not k.startswith("__")
    },
)


QuantizationDetails = enum.Enum(
    "QuantizationDetails",
    {
        k: v
        for k, v in vars(schema.QuantizationDetails).items()
        if not k.startswith("__")
    },
)


Padding = enum.Enum(
    "Padding", {k: v for k, v in vars(schema.Padding).items() if not k.startswith("__")}
)


BuiltinOptions = enum.Enum(
    "BuiltinOptions",
    {k: v for k, v in vars(schema.BuiltinOptions).items() if not k.startswith("__")},
)
