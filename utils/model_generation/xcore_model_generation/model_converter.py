# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod

import tensorflow as tf  # type: ignore

from .model_generator import ModelGenerator


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

        Particularly, this method is responsible for populating/modifying
        _converted_models of the owner.
        """
        raise NotImplementedError()


class TFLiteFloatConverter(ModelConverter):
    def convert(self) -> None:
        model_generator = self._model_generator
        model_generator._converted_models[
            self
        ] = tf.lite.TFLiteConverter.from_keras_model(model_generator._model).convert()


class TFLiteQuantConverter(ModelConverter):
    @abstractmethod
    def convert(self, *, repr_data=None, **kwargs) -> None:
        raise NotImplementedError()  # TODO:


class XCoreConverter(ModelConverter):
    @abstractmethod
    def convert(self, **kwargs) -> None:
        raise NotImplementedError()  # TODO:

