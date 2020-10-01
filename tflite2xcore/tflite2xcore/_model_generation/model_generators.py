# Copyright (c) 2020, XMOS Ltd, All rights reserved

import dill  # type: ignore
import logging
import tensorflow as tf  # type: ignore
from pathlib import Path
from abc import abstractmethod
from typing import Tuple, Union

from tflite2xcore.utils import set_all_seeds, LoggingContext  # type: ignore # TODO: fix this

from . import Configuration
from .runners import Runner, RunnerDependent


class ModelGenerator(RunnerDependent):
    """ Superclass for defining parameterized automatic model generation.

    The main use case of this class is generation of the integration test
    models.
    """

    _model: tf.keras.Model

    def __init__(self, runner: Runner) -> None:
        self._runner = runner

    @abstractmethod
    def build(self) -> None:
        """ Sets the _model field as needed by the subclass.
        
        The configuration should be set using the _set_config method before
        calling this.
        """
        raise NotImplementedError()

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
        self._model.save(dirpath / "model")
        tmp = self._model
        del self._model
        with open(dirpath / "generator.dill", "wb") as f:
            dill.dump(self, f)
        self._model = tmp
        return dirpath

    @classmethod
    def load(cls, dirpath: Union[Path, str]) -> "ModelGenerator":
        dirpath = Path(dirpath)
        with open(dirpath / "generator.dill", "rb") as f:
            obj = dill.load(f)
        assert isinstance(obj, cls)

        # tf may complain about missing training config, so silence it
        with LoggingContext(tf.get_logger(), logging.ERROR):
            obj._model = tf.keras.models.load_model(dirpath / "model")
        return obj
