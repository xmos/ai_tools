# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod
from typing import Optional

import tensorflow as tf  # type: ignore

from .model_generator import ModelGenerator, KerasModelGenerator, Configuration
from .utils import quantize_converter


class ModelConverter(ABC):
    """ Superclass for defining model conversion logic.

    ModelConverter objects are registered when a ModelGenerator object is
    instantiated.
    """

    def __init__(self, model_generator: ModelGenerator) -> None:
        """ Registers the ModelGenerator that owns this ModelConverter. """
        self._model_generator = model_generator

    @abstractmethod
    def convert(self) -> None:
        """ Mutates self._model_generator as defined in subclasses.

        Particularly, this method is responsible for populating/modifying the
        _converted_models field of the owner. This method should be called
        after the set_config method has prepared the converter.
        """
        raise NotImplementedError()

    def set_config(self) -> None:
        """ Configures the converter as needed by the owner. 
        
        By default no configuration is needed, but subclasses can implement
        this method as needed.
        """
        pass


class TFLiteFloatConverter(ModelConverter):
    """ Converts the _model field of a KerasModelGenerator to a floating point
    TFLite model.
    """

    _model_generator: KerasModelGenerator

    def convert(self) -> None:
        model_generator = self._model_generator
        model_generator._converted_models[
            self
        ] = tf.lite.TFLiteConverter.from_keras_model(model_generator._model).convert()


class TFLiteQuantConverter(ModelConverter):
    """ Converts the _model field of a KerasModelGenerator to a quantized
    TFLite model.
    """

    _model_generator: KerasModelGenerator
    _data_len: int
    _data_init: tf.initializers.Initializer

    def set_config(
        self,
        data_len: Optional[int] = None,
        data_init: Optional[tf.initializers.Initializer] = None,
    ) -> None:
        self._data_len = data_len or 10
        self._data_init = data_init or tf.random_uniform_initializer(-1, 1)

    def convert(self) -> None:
        model_generator = self._model_generator
        converter = tf.lite.TFLiteConverter.from_keras_model(model_generator._model)
        quantize_converter(
            converter,
            representative_data=self._data_init(
                (self._data_len, *model_generator._input_shape)
            ),
        )
        model_generator._converted_models[self] = converter.convert()


class XCoreConverter(ModelConverter):
    @abstractmethod
    def convert(self) -> None:
        raise NotImplementedError()  # TODO:

