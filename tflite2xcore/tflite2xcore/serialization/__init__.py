# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

from .xcore_schema import (
    QuantizationDetails,
    ActivationFunctionType,
    FullyConnectedOptionsWeightsFormat,
    Padding,
    TensorType,
    OperatorCode,
    BuiltinOpCodes,
    ExternalOpCodes,
    XCOREOpCodes,
    BuiltinOptions,
)
from .flatbuffers_c import FlexbufferParser

# TODO: remove this
def write_flatbuffer(model, filename):
    from tflite2xcore.xcore_model import XCOREModel

    assert isinstance(model, XCOREModel)
    return model.write_flatbuffer(filename)


# TODO: remove this
def read_flatbuffer(filename):
    from tflite2xcore.xcore_model import XCOREModel

    return XCOREModel.read_flatbuffer(filename)
