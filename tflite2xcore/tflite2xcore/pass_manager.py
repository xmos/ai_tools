# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import logging
from pathlib import Path
from collections import deque
from math import log10, ceil
from typing import TYPE_CHECKING, Iterable, Optional, List, Tuple, Deque

from tflite2xcore import tflite_visualize

if TYPE_CHECKING:
    from tflite2xcore.xcore_model import XCOREModel
    from tflite2xcore.transformation_passes import ModelTransformationPass


class PassManager:
    def __init__(
        self,
        model: Optional["XCOREModel"] = None,
        *,
        debug: bool = False,
        keep_intermediates: bool = False,
    ) -> None:
        self._queue: Deque["ModelTransformationPass"] = deque()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model: Optional["XCOREModel"] = None
        if model:
            self.register_model(model)
        self._mutating_passes: List[Tuple[int, str]] = []
        self._intermediates: List[bytes] = []
        self.keep_intermediates = keep_intermediates

    def register_model(self, model: "XCOREModel") -> None:
        assert model
        self._model = model

    @property
    def passes(self) -> Iterable["ModelTransformationPass"]:
        for trf_pass in self._queue:
            yield trf_pass

    def register_passes(self, other_mgr: "PassManager") -> None:
        for trf_pass in other_mgr.passes:
            self.register_pass(trf_pass)

    def register_pass(self, trf_pass: "ModelTransformationPass") -> None:
        self._queue.append(trf_pass)

    def pop_pass(self) -> "ModelTransformationPass":
        return self._queue.popleft()

    def save_intermediates(self, dirpath: Path, *, visualize: bool = True) -> None:
        if len(self._intermediates) == 0:
            self.logger.warning("No intermediate models were recorded!")

        dirpath.mkdir(parents=True, exist_ok=True)

        fill_width = ceil(log10(self._mutating_passes[-1][0]))
        for (j, _), bits in zip(self._mutating_passes, self._intermediates):
            basepath = dirpath.joinpath(
                f"model_{self.__class__.__name__}_{j:0{fill_width}d}"
            ).resolve()
            filepath = basepath.with_suffix(".tflite")
            with open(filepath, "wb") as f:
                f.write(bits)
            if visualize:
                tflite_visualize.main(filepath, basepath.with_suffix(".html"))
            self.logger.debug(f"Saved {filepath}")

    def run_passes(self) -> None:
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
