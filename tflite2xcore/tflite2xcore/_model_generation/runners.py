# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, NamedTuple, Optional, List

from . import Configuration

if TYPE_CHECKING:
    import numpy as np  # type: ignore
    from .evaluators import Evaluator
    from .converters import Converter
    from .model_generators import ModelGenerator


RunnerReports = Dict["Converter", Any]


class RunnerOutputs(NamedTuple):
    reference: "np.ndarray"
    xcore: "np.ndarray"


class Runner(ABC):
    """ Superclass for defining the behavior of model generation runs.

    A Runner registers a ModelGenerator object along with all the
    converters and evaluators.
    """

    reports: Optional[RunnerReports]
    outputs: Optional[RunnerOutputs]
    _config: Configuration

    def __init__(
        self,
        generator: "ModelGenerator",
        converters: Optional[List["Converter"]] = None,
        evaluators: Optional[List["Evaluator"]] = None,
    ) -> None:
        self._model_generator = generator
        self._converters = converters or []
        self._evaluators = evaluators or []

    @abstractmethod
    def __call__(self) -> None:
        """ Defines how self._model_generator should be run with a config.

        Optionally sets self.reports and self.outputs.
        """
        self.reports = None
        self.outputs = None

    def check_config(self) -> None:
        """ Checks if the current configuration parameters are legal. """
        # TODO: extend to converters and evaluators
        self._model_generator.check_config()

    def _set_config(self, cfg: Configuration) -> None:
        """ Sets the relevant configuration parameters.

        This method operates on the config input argument in-place.
        Subclasses should override this instead of the set_config method.
        """
        self._model_generator._set_config(cfg)
        for converter in self._converters:
            converter._set_config(cfg)

    def set_config(self, **config: Any) -> None:
        """ Configures the runner before it is called.
        
        Default values for missing configuration parameters are set.
        Subclasses should override set_config instead of this method.
        """
        self._config = {}
        self._set_config(config)
        if config:
            raise ValueError(
                f"Unexpected configuration parameter(s): {', '.join(config.keys())}"
            )
        self.check_config()


class RunnerDependent(ABC):
    def __init__(self, runner: "Runner") -> None:
        self._runner = runner

    @property
    def _config(self) -> Configuration:
        return self._runner._config

    def check_config(self) -> None:
        pass

    def _set_config(self, cfg: Configuration) -> None:
        """ Sets the relevant configuration parameters and returns the unused ones.
        
        This method operates on the cfg input argument in-place.
        """
        pass
