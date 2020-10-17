# Copyright (c) 2019-2020, XMOS Ltd, All rights reserved

# type: ignore

from enum import IntEnum

from . import schema_py_generated as schema


ActivationFunctionType = IntEnum(
    "ActivationFunctionType",
    {
        k: v
        for k, v in vars(schema.ActivationFunctionType).items()
        if not k.startswith("__")
    },
)


QuantizationDetails = IntEnum(
    "QuantizationDetails",
    {
        k: v
        for k, v in vars(schema.QuantizationDetails).items()
        if not k.startswith("__")
    },
)


FullyConnectedOptionsWeightsFormat = IntEnum(
    "FullyConnectedOptionsWeightsFormat",
    {
        k: v
        for k, v in vars(schema.FullyConnectedOptionsWeightsFormat).items()
        if not k.startswith("__")
    },
)

Padding = IntEnum(
    "Padding", {k: v for k, v in vars(schema.Padding).items() if not k.startswith("__")}
)
