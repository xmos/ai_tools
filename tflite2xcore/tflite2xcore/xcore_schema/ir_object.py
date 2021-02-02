# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, Sequence, List, Any, Optional

_S = TypeVar("_S", bound="_IRObject")

if TYPE_CHECKING:
    from .xcore_model import XCOREModel, Subgraph


class _IRObject(ABC):
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or ""

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


class _ModelDependent(_IRObject):
    _model: "XCOREModel"

    def __init__(
        self, name: Optional[str] = None, model: Optional["XCOREModel"] = None
    ):
        super().__init__(name)
        if model:
            model.register_dependent(self)

    @property
    def model(self) -> "XCOREModel":
        return self._model


class _SubgraphDependent(_IRObject):
    _subgraph: "Subgraph"

    @property
    def subgraph(self) -> "Subgraph":
        return self._subgraph

    @property
    def model(self) -> "XCOREModel":
        return self.subgraph._model
