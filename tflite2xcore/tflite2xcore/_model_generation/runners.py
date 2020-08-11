# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, NamedTuple, Optional

if TYPE_CHECKING:
    import numpy as np  # type: ignore
    from .converters import Converter
    from .model_generators import ModelGenerator


RunnerReports = Dict["Converter", Any]


class RunnerOutputs(NamedTuple):
    reference: "np.ndarray"
    xcore: "np.ndarray"


class Runner(ABC):
    """ Superclass for defining the behavior of model generation runs.

    A Runner is registered when a ModelGenerator object is instantiated.
    """

    reports: Optional[RunnerReports]
    outputs: Optional[RunnerOutputs]

    def __init__(self, generator: "ModelGenerator") -> None:
        """ Registers the generator that owns the runner. """
        self._model_generator = generator
        self.reports = None
        self.outputs = None

    @abstractmethod
    def __call__(self) -> None:
        """ Defines how self._model_generator should be run with a config.

        Optionally sets self.reports and self.outputs.
        """
        self._model_generator.build()
