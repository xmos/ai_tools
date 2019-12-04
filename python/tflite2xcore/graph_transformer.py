# Copyright (c) 2019, XMOS Ltd, All rights reserved

import enum
import heapq
import logging

from abc import ABC, abstractmethod
from .xcore_model import XCOREModel


class PassPriority(enum.IntEnum):
    def _generate_next_value_(name, start, count, last_values):  # pylint: disable=no-self-argument
        return last_values[-1] + 1 if last_values else 0

    # TODO: change these to meaningful names
    HIGHEST = enum.auto()
    HIGH = HIGHEST
    MEDIUM = enum.auto()
    LOW = enum.auto()
    LOWEST = LOW


class TransformationPass(ABC):
    def __init__(self, priority):
        assert isinstance(priority, PassPriority)
        self.priority = priority

    @abstractmethod
    def match(self, obj):
        pass

    @abstractmethod
    def mutate(self, obj):
        pass

    @abstractmethod
    def target_iterable(self, subgraph):
        pass

    def log_match(self, obj):
        logging.debug(f"{type(self).__name__} matched {obj}")

    def run_subgraph(self, subgraph):
        keep_running = True
        while keep_running:
            for obj in self.target_iterable(subgraph):
                if self.match(obj):
                    self.log_match(obj)
                    self.mutate(obj)
                    break
            else:
                keep_running = False

    def run(self, model):
        for j, subgraph in enumerate(model.subgraphs):
            logging.debug(f"{type(self).__name__} running on subgraph {j}")
            self.run_subgraph(subgraph)


class OperatorMatchingPass(TransformationPass):
    def __init__(self, priority):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.operators

    def log_match(self, op):
        super().log_match(f"operator {op.operator_code}")


class TensorMatchingPass(TransformationPass):
    def __init__(self, priority):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.tensors

    def log_match(self, tensor):
        super().log_match(f"tensor {tensor.name}")


class InputTensorMatchingPass(TransformationPass):
    def __init__(self, priority):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.inputs

    def log_match(self, tensor):
        super().log_match(f"input tensor {tensor.name}")


class OutputTensorMatchingPass(TransformationPass):
    def __init__(self, priority):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.outputs

    def log_match(self, tensor):
        super().log_match(f"output tensor {tensor.name}")


class PassManager():
    def __init__(self, model=None, passes=[]):
        self._queue = []
        if model:
            self.register_model(model)
        for trf_pass in passes:
            self.register_pass(trf_pass)

    def register_model(self, model):
        assert isinstance(model, XCOREModel)
        self._model = model

    def register_pass(self, trf_pass):
        assert isinstance(trf_pass, TransformationPass)
        heapq.heappush(self._queue, (trf_pass.priority, trf_pass))

    def run_passes(self):
        while self._queue:
            _, trf_pass = heapq.heappop(self._queue)
            logging.debug(f"running {trf_pass}...")
            trf_pass.run(self._model)
