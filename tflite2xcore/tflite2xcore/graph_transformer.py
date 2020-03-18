# Copyright (c) 2019, XMOS Ltd, All rights reserved

import enum
import heapq
import logging
import itertools

from abc import ABC, abstractmethod
from tflite2xcore.xcore_model import XCOREModel


class PassPriority(enum.IntEnum):
    def _generate_next_value_(name, start, count, last_values):  # pylint: disable=no-self-argument
        return last_values[-1] + 1 if last_values else 0

    # TODO: change these to meaningful names
    HIGHEST = enum.auto()
    PREP = HIGHEST
    HIGH = enum.auto()
    MEDIUM = enum.auto()
    FUSING = enum.auto()
    ARGMAX = enum.auto()
    PAR = enum.auto()
    CLEANUP = enum.auto()
    LOWEST = CLEANUP


class ModelTransformationPass(ABC):
    def __init__(self, priority):
        assert isinstance(priority, PassPriority)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.priority = priority

    @abstractmethod
    def run(self, model):
        pass

    def __str__(self):
        return self.__class__.__name__


class SubgraphTransformationPass(ModelTransformationPass):
    def __init__(self, priority, *, safe_mode=False):
        super().__init__(priority)
        self.safe_mode = safe_mode
        self.superseding_passes = []

    @abstractmethod
    def match(self, obj):
        if self.safe_mode:
            for p in self.superseding_passes:
                if p.match(obj):
                    return False

        return True

    @abstractmethod
    def mutate(self, obj):
        pass

    @abstractmethod
    def target_iterable(self, subgraph):
        pass

    def log_match(self, obj):
        self.logger.info(f"matched {obj}")

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
            self.logger.debug(f"running on subgraph {j}")
            self.run_subgraph(subgraph)


class OperatorMatchingPass(SubgraphTransformationPass):
    def __init__(self, priority):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.operators

    def log_match(self, op):
        super().log_match(f"operator {op.operator_code}")


class TensorMatchingPass(SubgraphTransformationPass):
    def __init__(self, priority):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.tensors

    def log_match(self, tensor):
        super().log_match(f"tensor {tensor.name}")


class InputTensorMatchingPass(SubgraphTransformationPass):
    def __init__(self, priority):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.inputs

    def log_match(self, tensor):
        super().log_match(f"input tensor {tensor.name}")


class OutputTensorMatchingPass(SubgraphTransformationPass):
    def __init__(self, priority):
        super().__init__(priority)

    def target_iterable(self, subgraph):
        return subgraph.outputs

    def log_match(self, tensor):
        super().log_match(f"output tensor {tensor.name}")


class PassManager():
    def __init__(self, model=None, passes=[]):
        self._queue = []
        self._counter = itertools.count()
        self.logger = logging.getLogger(self.__class__.__name__)
        if model:
            self.register_model(model)
        for trf_pass in passes:
            self.register_pass(trf_pass)

    def register_model(self, model):
        assert isinstance(model, XCOREModel)
        self._model = model

    def register_pass(self, trf_pass):
        assert isinstance(trf_pass, ModelTransformationPass)
        heapq.heappush(self._queue,
                       (trf_pass.priority, next(self._counter), trf_pass))

    def pop_pass(self):
        return heapq.heappop(self._queue)[-1]

    def run_passes(self):
        while self._queue:
            trf_pass = self.pop_pass()
            self.logger.debug(f"running {trf_pass}...")
            trf_pass.run(self._model)
