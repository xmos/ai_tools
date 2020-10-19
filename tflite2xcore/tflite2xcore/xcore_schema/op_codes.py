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
    @classmethod
    def add_new_opcode(cls, name: str) -> "ExternalOpCodes":
        assert name.isidentifier()
        try:
            return cls[name]
        except KeyError:
            aenum.extend_enum(cls, name)
            return cls[name]


ExternalOpCodes.add_new_opcode("LceQuantize")
ExternalOpCodes.add_new_opcode("LceBconv2d")
ExternalOpCodes.add_new_opcode("LceDequantize")


class XCOREOpCodes(enum.Enum):
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
