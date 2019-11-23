# Copyright (c) 2019, XMOS Ltd, All rights reserved

import enum
import heapq

from abc import ABC, abstractmethod


class PassPriority(enum.IntEnum):
    def _generate_next_value_(name, start, count, last_values):  # pylint: disable=no-self-argument
        return last_values[-1] + 1 if last_values else 0

    # TODO: change these to meaningful names
    HIGH = enum.auto()
    MEDIUM = enum.auto()
    LOW = enum.auto()


class TransformationPass(ABC):
    def __init__(self, priority):
        assert isinstance(priority, PassPriority)
        self.priority = priority

    @abstractmethod
    def match(self, op):
        pass

    @abstractmethod
    def mutate(self, op):
        pass

    def run_subgraph(self, subgraph):
        # TODO: assert that argument is valid subgraph object
        keep_running = True
        while keep_running:
            for op in subgraph.operators:
                if self.match(op):
                    # TODO: log a debug message here
                    self.mutate(op)
                    break
            else:
                keep_running = False

    def run(self, model):
        # TODO: assert that argument is valid model object
        for subgraph in model.subgraphs:
            self.run_subgraph(subgraph)


class PassManager():
    def __init__(self, model=None, passes=[]):
        self._queue = []
        if model:
            self.register_model(model)
        for trf_pass in passes:
            self.register_pass(trf_pass)

    def register_model(self, model):
        # TODO: assert that argument is valid model object
        self._model = model

    def register_pass(self, trf_pass):
        assert isinstance(trf_pass, TransformationPass)
        heapq.heappush(self._queue, (trf_pass.priority, trf_pass))

    def run_passes(self):
        while self._queue:
            _, trf_pass = heapq.heappop(self._queue)
            # TODO: log a debug message here
            trf_pass.run(self._model)
