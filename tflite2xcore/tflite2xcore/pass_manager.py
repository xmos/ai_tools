# Copyright (c) 2019, XMOS Ltd, All rights reserved

import logging
import pathlib
from typing import TYPE_CHECKING, Iterable
from collections import deque

from tflite2xcore import tflite_visualize

if TYPE_CHECKING:
    from tflite2xcore.xcore_model import XCOREModel
    from tflite2xcore.transformation_passes import ModelTransformationPass


class PassManager:
    def __init__(self, model=None, *, debug=False, keep_intermediates=False):
        self._queue = deque()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model = None
        if model:
            self.register_model(model)
        self._mutating_passes = []
        self._intermediates = []
        self.keep_intermediates = keep_intermediates

    def register_model(self, model: "XCOREModel"):
        self._model = model

    @property
    def passes(self) -> Iterable["ModelTransformationPass"]:
        for trf_pass in self._queue:
            yield trf_pass

    def register_passes(self, other_mgr: "PassManager") -> None:
        for trf_pass in other_mgr.passes:
            self.register_pass(trf_pass)

    def register_pass(self, trf_pass: "ModelTransformationPass"):
        self._queue.append(trf_pass)

    def pop_pass(self):
        return self._queue.popleft()

    def save_intermediates(self, dirpath, *, visualize=True):
        if len(self._intermediates) == 0:
            self.logger.warning("No intermediate models were recorded!")

        dirpath = pathlib.Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        for (j, _), bits in zip(self._mutating_passes, self._intermediates):
            basepath = dirpath.joinpath(
                f"model_{self.__class__.__name__}_{j}"
            ).resolve()
            filepath = basepath.with_suffix(".tflite")
            with open(filepath, "wb") as f:
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

            self.logger.debug(f"Running pass #{n}/{num_passes}: {trf_pass}..")

            modified = trf_pass.run(self._model)
            if __debug__:
                try:
                    self._model.sanity_check()
                except AssertionError as e:
                    self.logger.exception(e)

            if modified:
                self._mutating_passes.append((n, trf_pass.__class__.__name__))
                if self.keep_intermediates:
                    # switch descriptions for the intermediate models
                    new_desc = str(self._mutating_passes)
                    self._model.description, old_desc = (
                        new_desc,
                        self._model.description,
                    )
                    self._intermediates.append(self._model.serialize())
                    self._model.description = old_desc

        msg = "\n".join(
            [f"  #{p[0]}/{num_passes}: {p[1]}" for p in self._mutating_passes]
        )
        if msg:
            self.logger.info(f"The following passes mutated the model:\n{msg}")
