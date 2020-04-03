# Copyright (c) 2019, XMOS Ltd, All rights reserved

import enum
import heapq
import itertools

from abc import ABC, abstractmethod

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore import logging


class PassPriority(enum.IntEnum):
    def _generate_next_value_(name, start, count, last_values):  # pylint: disable=no-self-argument
        return last_values[-1] + 1 if last_values else 0

    HIGH = enum.auto()
    MEDIUM = enum.auto()
    LOW = enum.auto()


class ModelTransformationPass(ABC):
    def __init__(self, *, priority=PassPriority.MEDIUM, debug=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.priority = priority
        self.debug = debug

    @abstractmethod
    def run(self, model):
        pass

    def __str__(self):
        return self.__class__.__name__


class PassManager():
    def __init__(self, model=None, passes=[], *, debug=False):
        self._queue = []
        self.debug = debug
        self._counter = itertools.count()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model = None
        if model:
            self.register_model(model)
        for trf_pass in passes:
            self.register_pass(trf_pass)

    def register_model(self, model):
        assert isinstance(model, XCOREModel)
        self._model = model

    def register_pass(self, trf_pass):
        assert isinstance(trf_pass, ModelTransformationPass)
        trf_pass.debug = trf_pass.debug or self.debug
        heapq.heappush(self._queue,
                       (trf_pass.priority, next(self._counter), trf_pass))

    def pop_pass(self):
        return heapq.heappop(self._queue)[-1]

    def run_passes(self):
        if not self._model:
            raise Exception("No model registered!")

        self.logger.debug(f"Running {len(self._queue)} passes...")
        cnt = 0
        while self._queue:
            trf_pass = self.pop_pass()

            self.logger.debug(f"Running pass #{cnt}: "
                              f"{trf_pass} (priority={trf_pass.priority.name})...")
            if self.debug:
                import pdb; pdb.set_trace()

            trf_pass.run(self._model)
            cnt += 1
