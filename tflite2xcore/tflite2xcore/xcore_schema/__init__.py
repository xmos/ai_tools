# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod
from typing import TypeVar, Sequence, List, Any, Optional

_S = TypeVar("_S", bound="_IRObject")


class _IRObject(ABC):
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name

    @abstractmethod
    def sanity_check(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def sequence_equal(l1: Sequence[_S], l2: Sequence[_S]) -> bool:
        return len(l1) == len(l2) and all(a.is_equal(b) for a, b in zip(l1, l2))

    @staticmethod
    def _remove_if_contained(ll: List[_S], obj: _S) -> None:
        try:
            ll.remove(obj)
        except ValueError:
            pass

    def is_equal(self, other: Any) -> bool:
        if type(self) is type(other):
            self.sanity_check()
            other.sanity_check()
            return True
        return False


from . import flexbuffers
from .tensor_type import TensorType
from .buffer import Buffer, _BufferDataType, _BufferOwnerContainer, _T
from .operator_codes import (
    OperatorCode,
    ValidOpCodes,
    CustomOpCodes,
    BuiltinOpCodes,
    ExternalOpCodes,
    XCOREOpCodes,
)

from .xcore_schema import (
    QuantizationDetails,
    ActivationFunctionType,
    FullyConnectedOptionsWeightsFormat,
    Padding,
    BuiltinOptions,
)

from .xcore_model import Buffer, Tensor, Operator, Subgraph, Metadata, XCOREModel

from . import xcore_model
