# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
from abc import abstractmethod
from typing import Tuple, Optional, Any

from . import Configuration, Hook
from .utils import parse_init_config
from .runners import Runner, RunnerDependent


class DataFactory(RunnerDependent):
    @abstractmethod
    def make_data(self) -> tf.Tensor:
        raise NotImplementedError()


class InitializerDataFactory(DataFactory):
    def __init__(self, runner: Runner, shape_hook: Hook[Tuple[int, ...]]):
        super().__init__(runner)
        self._shape_hook = shape_hook

    @property
    @abstractmethod
    def initializer(self) -> tf.keras.initializers.Initializer:
        raise NotImplementedError()

    def make_data(self, batch: Optional[int] = None) -> tf.Tensor:
        shape = self._shape_hook()
        if batch is not None:
            shape = (batch, *shape)

        return self.initializer(shape)


class InputInitializerDataFactory(InitializerDataFactory):
    def _set_config(self, cfg: Configuration) -> None:
        if "input_init" not in self._config:
            self._config["input_init"] = cfg.pop("input_init", ("RandomUniform", -1, 1))
        super()._set_config(cfg)

    @property
    def initializer(self) -> tf.keras.initializers.Initializer:
        return parse_init_config(*self._config["input_init"])
