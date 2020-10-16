# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

from . import flexbuffers
from .xcore_schema import (
    QuantizationDetails,
    ActivationFunctionType,
    FullyConnectedOptionsWeightsFormat,
    Padding,
    TensorType,
    OperatorCode,
    ValidOpCodes,
    BuiltinOpCodes,
    ExternalOpCodes,
    XCOREOpCodes,
    BuiltinOptions,
)

from .xcore_model import Buffer, Tensor, Operator, Subgraph, Metadata, XCOREModel

from . import xcore_model
