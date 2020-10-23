# Copyright (c) 2020, XMOS Ltd, All rights reserved

# type: ignore

import enum
import aenum
from typing import Union

from . import schema_py_generated as schema

BuiltinOpCodes = enum.Enum(
    "BuiltinOpCodes",
    {k: v for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("__")},
)


class ExternalOpCodes(aenum.Enum):
    def _generate_next_value_(name: str, *_) -> str:  # pylint: disable=no-self-argument
        return name

    LceQuantize = aenum.auto()
    LceBconv2d = aenum.auto()
    LceDequantize = aenum.auto()

    @classmethod
    def add_new_opcode(cls, name: str) -> "ExternalOpCodes":
        assert name.isidentifier()
        try:
            return cls[name]
        except KeyError:
            aenum.extend_enum(cls, name)
            return cls[name]


class XCOREOpCodes(enum.Enum):
    def _generate_next_value_(name: str, *_) -> str:  # pylint: disable=no-self-argument
        return name

    XC_lookup_8 = enum.auto()
    XC_argmax_16 = enum.auto()  # currently not used by any passes
    XC_maxpool2d = enum.auto()
    XC_avgpool2d = enum.auto()
    XC_avgpool2d_global = enum.auto()
    XC_fc = enum.auto()
    XC_requantize_16_to_8 = enum.auto()  # currently unused
    XC_conv2d_shallowin = enum.auto()
    XC_conv2d_deep = enum.auto()
    XC_conv2d_1x1 = enum.auto()
    XC_conv2d_depthwise = enum.auto()
    XC_bsign_8 = enum.auto()
    XC_bconv2d_int8 = enum.auto()
    XC_bconv2d_int8_DIDO = enum.auto()
    XC_bconv2d_bin = enum.auto()
    XC_bconv2d_bin_DI = enum.auto()
