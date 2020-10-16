# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

from . import flexbuffers
from .tensor_type import TensorType
from .xcore_schema import (
    QuantizationDetails,
    ActivationFunctionType,
    FullyConnectedOptionsWeightsFormat,
    Padding,
    OperatorCode,
    ValidOpCodes,
    BuiltinOpCodes,
    ExternalOpCodes,
    XCOREOpCodes,
    BuiltinOptions,
)

from .xcore_model import Buffer, Tensor, Operator, Subgraph, Metadata, XCOREModel

from . import xcore_model
