# Copyright (c) 2020, XMOS Ltd, All rights reserved

import dill  # type: ignore
import logging
import tensorflow as tf  # type: ignore
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any, Union

from tflite2xcore.utils import set_all_seeds, LoggingContext  # type: ignore # TODO: fix this

from . import Configuration
from .converters import Converter
from .runners import Runner
from .evaluators import Evaluator


class ModelGenerator(ABC):
    """ Superclass for defining parameterized automatic model generation.

    The main use case of this class is generation of the integration test
    models.
    """

    _model: Any
    _config: Configuration
    run: Runner

    def __init__(
        self,
        runner: Runner,
        converters: Optional[List[Converter]] = None,
        evaluators: Optional[List[Evaluator]] = None,
    ) -> None:
        """ Registers the runner, converters and evaluators. """
        self.run = runner
        self._converters = converters or []
        self._evaluators = evaluators or []

    @abstractmethod
    def build(self) -> None:
        """ Sets the _model field as needed by the subclass.
        
        The configuration should be set using the set_config method before
        calling this.
        """
        raise NotImplementedError()

    @abstractmethod
    def _set_config(self, cfg: Configuration) -> None:
        """ Sets the relevant configuration parameters and returns the unused ones.
        
        Should check if the given configuration parameters are legal.
        This method operates on the config input argument in-place.
        Subclasses should implement this instead of the set_config method.
        """
        for converter in self._converters:
            converter._set_config(cfg)

    def set_config(self, **config: Any) -> None:
        """ Configures the model generator before the build method is run.
        
        Default values for missing configuration parameters are set.
        Subclasses should implement the _set_config method instead of this.
        """
        self._config = {}
        self._set_config(config)
        if config:
            raise ValueError(
                f"Unexpected configuration parameter(s): {', '.join(config.keys())}"
            )


class KerasModelGenerator(ModelGenerator):
    _model: tf.keras.Model

    def _prep_backend(self) -> None:
        tf.keras.backend.clear_session()
        set_all_seeds()

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._model.input_shape[1:]  # type:ignore  # pylint: disable=no-member

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self._model.output_shape[1:]  # type:ignore  # pylint: disable=no-member

    def save(self, dirpath: Union[Path, str]) -> Path:
        """ Saves the model contents to the specified directory.
        
        If the directory doesn't exist, it is created.
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        self._model.save(dirpath / "model.h5")
        tmp = self._model
        del self._model
        with open(dirpath / "generator.dill", "wb") as f:
            dill.dump(self, f)
        self._model = tmp
        return dirpath

    @classmethod
    def load(cls, dirpath: Union[Path, str]) -> "KerasModelGenerator":
        dirpath = Path(dirpath)
        with open(dirpath / "generator.dill", "rb") as f:
            obj = dill.load(f)
        assert isinstance(obj, cls)

        # tf may complain about missing training config, so silence it
        with LoggingContext(tf.get_logger(), logging.ERROR):
            obj._model = tf.keras.models.load_model(dirpath / "model.h5")
        return obj
