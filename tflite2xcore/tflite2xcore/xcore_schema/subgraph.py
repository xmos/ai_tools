# Copyright 2019-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Iterable, List

from . import (
    _ModelDependent,
    TensorType,
    OperatorCode,
    Buffer,
    _ShapeInputType,
    Tensor,
    Operator,
    _OpOptionsType,
)

if TYPE_CHECKING:
    from . import XCOREModel


class Subgraph(_ModelDependent):
    def __init__(
        self,
        name: Optional[str] = None,
        model: Optional["XCOREModel"] = None,
        inputs: Optional[Iterable[Tensor]] = None,
        outputs: Optional[Iterable[Tensor]] = None,
        operators: Optional[Iterable[Operator]] = None,
        tensors: Optional[Iterable[Tensor]] = None,
    ) -> None:
        super().__init__(name, model)
        self.inputs: List[Tensor] = list(inputs or [])
        self.outputs: List[Tensor] = list(outputs or [])
        self.operators: List[Operator] = list(operators or [])
        self.tensors: List[Tensor] = list(tensors or [])

    @property
    def intermediates(self) -> List[Tensor]:
        # intermediates are any tensors that are not an input or an output
        return [t for t in self.tensors if t not in (self.inputs + self.outputs)]

    def is_equal(self, other: Any) -> bool:
        return (
            super().is_equal(other)
            # and self.name == other.name  # intentionally not compared
            and self.sequence_equal(self.inputs, other.inputs)
            and self.sequence_equal(self.outputs, other.outputs)
            and self.sequence_equal(self.operators, other.operators)
            and self.sequence_equal(self.tensors, other.tensors)
        )

    def create_tensor(
        self,
        name: str,
        type_: TensorType,
        shape: _ShapeInputType,
        *,
        buffer: Optional[Buffer] = None,
        quantization: Optional[_OpOptionsType] = None,
        isinput: bool = False,
        isoutput: bool = False,
        producers: Optional[Iterable[Operator]] = None,
        consumers: Optional[Iterable[Operator]] = None,
    ) -> Tensor:

        name = self.make_unique_tensor_name(name)
        tensor = Tensor(self, name, type_, shape, quantization, producers, consumers)

        if buffer is None:
            try:
                buffer = Buffer(self._model)
            except AttributeError:
                buffer = Buffer()
        tensor.buffer = buffer
        tensor.buffer.owners.append(tensor)

        self.tensors.append(tensor)
        if isinput:
            self.inputs.append(tensor)
        if isoutput:
            self.outputs.append(tensor)
        return tensor

    def remove_tensor(self, tensor: Tensor) -> None:
        """ Removes the tensor from the subgraph and cuts all its connections.

            Note that the tensor will be left in an illegal state.
        """
        assert tensor in self.tensors
        self.tensors.remove(tensor)
        self._remove_if_contained(self.inputs, tensor)
        self._remove_if_contained(self.outputs, tensor)
        for op in tensor.consumers:
            self._remove_if_contained(op.inputs, tensor)
        for op in tensor.producers:
            self._remove_if_contained(op.outputs, tensor)
        tensor.buffer.owners.remove(tensor)

        # TODO: fix this
        # del tensor.consumers, tensor.producers, tensor.subgraph, tensor.buffer
        del tensor.consumers, tensor.producers, tensor.buffer

    def generate_unique_op_name(self, operator_code: OperatorCode) -> str:
        existing_names = [op.name for op in self.operators]
        j = 0
        while True:
            j, new_name = j + 1, f"{operator_code.name}_{j}"
            if new_name not in existing_names:
                return new_name

    def make_unique_tensor_name(self, candidate_name: str) -> str:
        existing_names = [
            name
            for tensor in self.tensors
            for name in (tensor.name, tensor.sanitized_name)
        ]

        j, new_name = 1, candidate_name
        while True:
            if new_name not in existing_names:
                return new_name
            j, new_name = j + 1, f"{candidate_name}_{j}"

    def create_operator(
        self,
        operator_code: OperatorCode,
        *,
        inputs: Optional[Iterable[Tensor]] = None,
        outputs: Optional[Iterable[Tensor]] = None,
        builtin_options: Optional[_OpOptionsType] = None,
        custom_options: Optional[_OpOptionsType] = None,
    ) -> Operator:
        name = self.generate_unique_op_name(operator_code)
        operator = Operator(
            self, operator_code, name, inputs, outputs, builtin_options, custom_options,
        )
        self.operators.append(operator)
        for input_tensor in operator.inputs:
            input_tensor.consumers.append(operator)
        for output_tensor in operator.outputs:
            output_tensor.producers.append(operator)
        return operator

    def remove_operator(self, op: Operator) -> None:
        """ Removes the operator from the subgraph and cuts all its connections.

            Note that the operator will be left in an illegal state.
        """
        assert op in self.operators
        self.operators.remove(op)
        for t in op.inputs:
            self._remove_if_contained(t.consumers, op)
        for t in op.outputs:
            self._remove_if_contained(t.producers, op)
        del op.inputs, op.outputs, op.subgraph

    def insert_operator(
        self, ref_op: Operator, new_op: Operator, after: bool = False
    ) -> None:
        """NOTE: this does not rewire inputs/outputs"""
        # find location of reference op
        try:
            ref_idx = self.operators.index(ref_op)
        except ValueError as e:
            raise ValueError("Cannot find reference operator in the subgraph") from e

        # remove new_op from list if already in the subgraph
        if new_op in self.operators:
            self.operators.remove(new_op)

        # (re)insert new op before/after reference op
        self.operators.insert(ref_idx + (1 if after else 0), new_op)

    def replace_operator(self, op: Operator, new_op: Operator) -> None:
        """NOTE: this does not rewire inputs/outputs"""
        # insert new op
        try:
            self.insert_operator(op, new_op)
        except ValueError:
            raise ValueError("Cannot find operator to replace in the subgraph")
        # remove old op
        self.remove_operator(op)

    def clone_tensor(self, tensor: Tensor) -> Tensor:
        return self.create_tensor(
            tensor.name,
            tensor.type,
            tensor.shape,
            quantization=deepcopy(tensor.quantization),
            buffer=Buffer(self.model, tensor.buffer.data),
        )

    def get_tensor(self, name: str) -> Tensor:
        for t in self.tensors:
            if t.name == name:
                return t
        raise ValueError(f"Tensor with name {name} not found!")

    def sanity_check(self) -> None:
        assert self in self.model.subgraphs
        # check for duplicates
        assert len(self.inputs) == len(set(self.inputs))
        assert len(self.outputs) == len(set(self.outputs))
        assert len(self.operators) == len(set(self.operators))
        assert len(self.tensors) == len(set(self.tensors))
        # make sure inputs and outputs are not misplaced
        for tensor in self.inputs + self.outputs:
            assert tensor in self.tensors
        # the subgraph is sane as long as all its objects are sane
        for op in self.operators:
            op.sanity_check()
        for tensor in self.tensors:
            tensor.sanity_check()
