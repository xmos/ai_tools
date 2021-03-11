# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from abc import ABC, abstractmethod
from collections import deque
from typing import Sequence, Dict

from tflite2xcore import xcore_schema as xir


class ExecutionPlanner(ABC):
    def __init__(self, subgraph: xir.Subgraph):
        self._graph = subgraph

    @abstractmethod
    def make_plan(self) -> Sequence[xir.Operator]:
        raise NotImplementedError()


class ReverseDepthFirstPlanner(ExecutionPlanner):
    def make_plan(self) -> Sequence[xir.Operator]:
        # rely on dict's insertion order guarantee (CPython 3.6+)
        reverse_op_order: Dict[xir.Operator, None] = {}

        # initialize the op stack with a sentinel that we'll remove later
        sentinel_op = self._graph.create_operator(
            xir.OperatorCode(xir.XCOREOpCodes.DUMMY), inputs=self._graph.outputs,
        )
        sentinel_op.name = "SENTINEL"
        op_stack = [sentinel_op]

        # dependency counts to be used to resolve ops that have multiple consumers
        dependency_counts: Dict[xir.Operator, int] = {sentinel_op: 1}

        while op_stack:
            op = op_stack.pop(-1)
            if op in reverse_op_order:
                # op already scheduled
                continue

            if op not in dependency_counts:
                # this is the first time we see this op, so count the dependencies
                dependency_counts[op] = len(
                    {c for t in op.outputs for c in t.consumers}
                )

            if dependency_counts[op] <= 0:
                raise Exception(
                    "Found operator with 0 or fewer dependencies (the graph may be corrupted)"
                )

            dependency_counts[op] -= 1
            if dependency_counts[op]:
                # skip scheduling of op if there are other dependents
                continue

            reverse_op_order[op] = None
            for tin in sorted(op.inputs, key=lambda t: t.size):
                op_stack.extend(tin.producers)

        # remove sentinel op
        self._graph.remove_operator(sentinel_op)
        del reverse_op_order[sentinel_op]

        # return ops in reverse order
        return list(reversed(list(reverse_op_order.keys())))

