# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from typing import TYPE_CHECKING, Optional, Dict, Any, Iterable

if TYPE_CHECKING:
    from . import Tensor, Subgraph, XCOREModel

from . import _IRObject, OperatorCode


_OpOptionsType = Dict[str, Any]


class Operator(_IRObject):
    name: str

    def __init__(
        self,
        subgraph: "Subgraph",
        operator_code: OperatorCode,
        name: Optional[str] = None,
        inputs: Optional[Iterable["Tensor"]] = None,
        outputs: Optional[Iterable["Tensor"]] = None,
        builtin_options: Optional[_OpOptionsType] = None,
        custom_options: Optional[_OpOptionsType] = None,
    ) -> None:
        # Generally, do not use this constructor to instantiate Operator!
        # Use Subgraph.create_operator instead.

        super().__init__(name or "")
        self.subgraph = subgraph  # parent
        self.operator_code = operator_code
        self.inputs = list(inputs or [])
        self.outputs = list(outputs or [])
        self.builtin_options = builtin_options or {}
        self.custom_options = custom_options or {}

    def add_custom_options(self, **kwargs: Any) -> None:
        if kwargs:
            self.custom_options.update(kwargs)

    @property
    def model(self) -> "XCOREModel":
        return self.subgraph.model

    def __str__(self) -> str:
        return f"({self.subgraph.operators.index(self)}) operator_code={self.operator_code}"

    def is_equal(self, other: Any) -> bool:
        return (
            super().is_equal(other)
            and self.operator_code == other.operator_code
            # and self.name == other.name  # intentionally not compared
            and self.sequence_equal(self.inputs, other.inputs)
            and self.sequence_equal(self.outputs, other.outputs)
            and self.builtin_options == other.builtin_options
            and self.custom_options == other.custom_options
        )

    def sanity_check(self) -> None:
        assert self in self.subgraph.operators
        # check double links with inputs/outputs
        for tensor in self.inputs:
            assert self in tensor.consumers
        for tensor in self.outputs:
            assert self in tensor.producers
