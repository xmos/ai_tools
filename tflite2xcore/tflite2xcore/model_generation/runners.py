# Copyright (c) 2020, XMOS Ltd, All rights reserved

import dill  # type: ignore
import numpy as np  # type: ignore
from pathlib import Path
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, NamedTuple, Optional, List, Union, Type

from . import Configuration

if TYPE_CHECKING:
    from .evaluators import Evaluator
    from .converters import Converter
    from .data_factories import DataFactory
    from .model_generators import ModelGenerator


RunnerReports = Dict["Converter", Any]


class RunnerOutputs(NamedTuple):
    reference: np.ndarray
    xcore: np.ndarray


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
        generator: Type["ModelGenerator"],
        converters: Optional[List["Converter"]] = None,
        evaluators: Optional[List["Evaluator"]] = None,
        data_factories: Optional[List["DataFactory"]] = None,
    ) -> None:
        self._model_generator = generator(self)
        self._converters = converters or []
        self._evaluators = evaluators or []
        self._data_factories = data_factories or []

    @abstractmethod
    def run(self) -> None:
        """ Defines how self._model_generator should be run with a config.

        Optionally sets self.reports and self.outputs.
        """
        self._model_generator.build()
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
        for evaluator in self._evaluators:
            evaluator._set_config(cfg)
        for data_factory in self._data_factories:
            data_factory._set_config(cfg)

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

    def save(self, dirpath: Union[Path, str]) -> Path:
        """ Saves the Runner contents to the specified directory.
        
        If the directory doesn't exist, it is created.
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        with self._model_generator.save_model(dirpath):
            with open(dirpath / "runner.dill", "wb") as f:
                dill.dump(self, f)

        return dirpath

    @classmethod
    def load(cls, dirpath: Union[Path, str]) -> "Runner":
        dirpath = Path(dirpath)
        with open(dirpath / "runner.dill", "rb") as f:
            obj: "Runner" = dill.load(f)
        assert isinstance(obj, cls)

        obj._model_generator.load_model(dirpath)
        return obj


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
