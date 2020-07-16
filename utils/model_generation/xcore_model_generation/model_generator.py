# Copyright (c) 2020, XMOS Ltd, All rights reserved

from inspect import isabstract
from typing import Union, List, Dict, Any, TYPE_CHECKING
from abc import ABC, abstractmethod

import tensorflow as tf  # type: ignore

if TYPE_CHECKING:
    from .model_converter import ModelConverter


Configuration = Dict[str, Any]


class ModelGenerator(ABC):
    """ Superclass for defining parameterized automatic model generation.

    The main use case of this class is generation of the integration test
    models.
    """

    _model: Union[tf.keras.Model, None] = None
    _converted_models: Dict["ModelConverter", Any] = {}
    _converters: List["ModelConverter"]

    def __init__(self) -> None:
        """ Registers the converters associated with the generated models. """
        self._converters = []

    @classmethod
    def builtin_configs(cls) -> List[Configuration]:
        """ Returns the basic configurations the build method should be run with.

        This method can load the configuration of a .yml file or defined in
        the function body.
        """
        return []

    @classmethod
    def runnable_subclasses(cls) -> List[type]:
        """ Returns a list of non-abstract child classes including the class itself. """
        return [
            class_ for class_ in [cls] + cls.__subclasses__() if not isabstract(class_)
        ]

    @abstractmethod
    def build(self, **cfg) -> None:
        """ Sets the _model field with a tf.keras.Model object.
        
        Generally, it should check the configuration using the check_config
        method before building the keras model. When this method is run, it
        should also reset the _converted_models field.
        """
        self.check_config(**cfg)
        self._converted_models = {}

    @classmethod
    @abstractmethod
    def check_config(cls, **cfg) -> None:
        """ Check if the given configuration parameters are legal."""
        pass

    @classmethod
    def parse_cfg(cls, cfg_string: str) -> List[Configuration]:
        """ Parses one or multiple lines of configuration string (e.g. yml). """
        raise NotImplementedError()


class TestModelGenerator(ModelGenerator):
    @classmethod
    @abstractmethod
    def builtin_configs(cls, level: str = "default") -> List[Configuration]:
        return []
