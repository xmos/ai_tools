# Copyright (c) 2019, XMOS Ltd, All rights reserved

import enum
import heapq
import itertools

from abc import ABC, abstractmethod

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore import logging, tflite_visualize
from tflite2xcore.serialization import serialize_model
from tflite2xcore.utils import convert_path


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
        return 0

    def __str__(self):
        return self.__class__.__name__


class PassManager():
    def __init__(self, model=None, passes=[], *,
                 debug=False, keep_intermediates=True):
        self._queue = []
        self.debug = debug
        self._counter = itertools.count()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model = None
        if model:
            self.register_model(model)
        for trf_pass in passes:
            self.register_pass(trf_pass)
        self._mutating_passes = []
        self._intermediates = []
        self.keep_intermediates = keep_intermediates

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

    def save_intermediates(self, dirpath, *, visualize=True):
        if len(self._intermediates) == 0:
            self.logger.warning("No intermediate models states were recorded!")
            return

        dirpath = convert_path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        for (j, _), bits in zip(self._mutating_passes, self._intermediates):
            basepath = dirpath.joinpath(f"model_{j}").resolve()
            filepath = basepath.with_suffix(".tflite")
            with open(filepath, 'wb') as f:
                f.write(bits)
            if visualize:
                tflite_visualize.main(filepath, basepath.with_suffix(".html"))
            self.logger.debug(f"Saved {filepath}")

    def run_passes(self):
        if not self._model:
            raise Exception("No model registered!")

        num_passes = len(self._queue)
        self.logger.debug(f"Running {num_passes} passes...")
        for n in range(num_passes):
            trf_pass = self.pop_pass()

            self.logger.debug(f"Running pass #{n}/{num_passes}: "
                              f"{trf_pass} (priority={trf_pass.priority.name})...")
            if self.debug:
                import pdb; pdb.set_trace()

            modified = trf_pass.run(self._model)
            if self.debug:
                try:
                    self._model.sanity_check()
                except AssertionError as e:
                    self.logger.exception(e)

            if modified:
                self._mutating_passes.append((n, trf_pass.__class__.__name__))
                if self.keep_intermediates:
                    # switch descriptions for the intermediate models
                    new_desc = str(self._mutating_passes)
                    self._model.description, old_desc = new_desc, self._model.description
                    self._intermediates.append(serialize_model(self._model))
                    self._model.description = old_desc

        msg = '\n'.join([
            f"  #{p[0]}/{num_passes}: {p[1]}" for p in self._mutating_passes
        ])
        self.logger.info(f"The following passes mutated the model:\n{msg}")
