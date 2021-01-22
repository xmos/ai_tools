# Copyright (c) 2019-2020, XMOS Ltd, All rights reserved

from enum import IntEnum

# TODO: consider adding fields to this enums for IDE support

class ActivationFunctionType(IntEnum):
    NONE: ActivationFunctionType
    RELU: ActivationFunctionType
    RELU_N1_TO_1: ActivationFunctionType
    RELU6: ActivationFunctionType
    TANH: ActivationFunctionType
    SIGN_BIT: ActivationFunctionType

class QuantizationDetails(IntEnum): ...
class FullyConnectedOptionsWeightsFormat(IntEnum): ...
class Padding(IntEnum): ...

