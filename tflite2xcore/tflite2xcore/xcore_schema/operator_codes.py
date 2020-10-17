# Copyright (c) 2020, XMOS Ltd, All rights reserved

import enum
import aenum
from typing import Optional, Union, Any

from . import schema_py_generated as schema

BuiltinOpCodes = enum.IntEnum(  # type: ignore
    "BuiltinOpCodes",
    {k: v for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("__")},
)


class ExternalOpCodes(aenum.Enum):  # type: ignore
    LceQuantize = "LceQuantize"
    LceBconv2d = "LceBconv2d"
    LceDequantize = "LceDequantize"

    @classmethod
    def add_new_opcode(cls, name: str) -> "ExternalOpCodes":
        assert name.isidentifier()
        try:
            return cls[name]  # type: ignore
        except KeyError:
            aenum.extend_enum(cls, name)
            return cls[name]  # type: ignore


class XCOREOpCodes(enum.Enum):
    # TODO: consider an IntEnum for this instead of strings
    XC_lookup_8 = "XC_lookup_8"
    XC_argmax_16 = "XC_argmax_16"  # currently not used by any passes
    XC_maxpool2d = "XC_maxpool2d"
    XC_avgpool2d = "XC_avgpool2d"
    XC_avgpool2d_global = "XC_avgpool2d_global"
    XC_fc = "XC_fc"
    XC_requantize_16_to_8 = "XC_requantize_16_to_8"  # currently unused
    XC_conv2d_shallowin = "XC_conv2d_shallowin"
    XC_conv2d_deep = "XC_conv2d_deep"
    XC_conv2d_1x1 = "XC_conv2d_1x1"
    XC_conv2d_depthwise = "XC_conv2d_depthwise"
    XC_bsign_8 = "XC_bsign_8"
    XC_bconv2d_int8 = "XC_bconv2d_int8"
    XC_bconv2d_int8_DIDO = "XC_bconv2d_int8_DIDO"
    XC_bconv2d_bin = "XC_bconv2d_bin"
    XC_bconv2d_bin_DI = "XC_bconv2d_bin_DI"


CustomOpCodes = Union[XCOREOpCodes, ExternalOpCodes]

ValidOpCodes = Union[BuiltinOpCodes, CustomOpCodes]


class OperatorCode:
    def __init__(self, opcode: ValidOpCodes, *, version: Optional[int] = None) -> None:
        self.version = version or 1
        self.code = opcode

    @property
    def name(self) -> str:
        return self.code.name

    @property
    def value(self) -> Union[int, str]:
        return self.code.value

    def __eq__(self, obj: Any) -> bool:
        return (
            isinstance(obj, OperatorCode)
            and obj.code is self.code
            and obj.version == self.version
        )

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return f"{self.code} (version {self.version})"
