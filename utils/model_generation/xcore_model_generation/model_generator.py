# Copyright (c) 2020, XMOS Ltd, All rights reserved

from inspect import isabstract
from typing import Optional, Tuple, List, Dict, Any
from abc import ABC, abstractmethod

import tensorflow as tf  # type: ignore

from tflite2xcore.utils import set_all_seeds  # type: ignore # TODO: fix this

from .model_converter import (
    ModelConverter,
    TFLiteFloatConverter,
    TFLiteQuantConverter,
    XCoreConverter,
)


Configuration = Dict[str, Any]


class ModelGenerator(ABC):
    """ Superclass for defining parameterized automatic model generation.

    The main use case of this class is generation of the integration test
    models.
    """

    _model: tf.keras.Model
    _config: Configuration = {}
    _converted_models: Dict[ModelConverter, Any] = {}

    def __init__(self, converters: Optional[List[ModelConverter]] = None) -> None:
        """ Registers the converters associated with the generated models. """
        self._converters = converters or []

    @classmethod
    def builtin_configs(cls) -> List[Configuration]:
        """ Returns the basic configurations the build method should be run with.

        This method can load the configuration of a .yml file or defined in
        the function body.
        """
        return []

    @abstractmethod
    def build(self) -> None:
        """ Sets the _model field with a tf.keras.Model object.
        
        The configuration should be set using the set_config method before
        calling this. When this method is run, it should also reset the
        _converted_models field.
        """
        self._converted_models = {}

    @abstractmethod
    def set_config(self, **config: Any) -> None:
        """ Configures the model generator before the build method is run.
        
        Should check if the given configuration parameters are legal.
        Optionally sets the default values for missing configuration parameters.
        """
        self._config = config

    @classmethod
    def parse_config(cls, config_string: str) -> List[Configuration]:
        """ Parses one or multiple lines of configuration string (e.g. yml). """
        raise NotImplementedError()


class KerasModelGenerator(ModelGenerator):
    def _prep_backend(self) -> None:
        tf.keras.backend.clear_session()
        set_all_seeds()

    @property
    def _input_shape(self) -> Tuple[int, ...]:
        return self._model.input_shape[1:]  # type:ignore  # pylint: disable=no-member

    @property
    def _output_shape(self) -> Tuple[int, ...]:
        return self._model.output_shape[1:]  # type:ignore  # pylint: disable=no-member


class IntegrationTestModelGenerator(KerasModelGenerator):
    _quant_converter: TFLiteQuantConverter
    _xcore_converter: XCoreConverter

    def __init__(self) -> None:
        self._quant_converter = TFLiteQuantConverter(self)
        self._xcore_converter = XCoreConverter(self, self._quant_converter)
        super().__init__(
            converters=[
                TFLiteFloatConverter(self),
                self._quant_converter,
                self._xcore_converter,
            ]
        )

    @classmethod
    @abstractmethod
    def builtin_configs(cls, level: str = "default") -> List[Configuration]:
        return []
