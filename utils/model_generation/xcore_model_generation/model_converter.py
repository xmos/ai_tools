# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Any, Callable, ByteString

import tensorflow as tf  # type: ignore

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.converter import optimize_for_xcore  # type: ignore # TODO: fix this

from .utils import quantize_converter

if TYPE_CHECKING:
    from .model_generator import (
        ModelGenerator,
        KerasModelGenerator,
        IntegrationTestModelGenerator,
        Configuration,
    )


TFLiteModel = ByteString


class ModelConverter(ABC):
    """ Superclass for defining model conversion logic and storing converted models.

    ModelConverter objects are registered when a ModelGenerator object is
    instantiated.
    """

    _model: TFLiteModel

    def __init__(self, model_generator: "ModelGenerator") -> None:
        """ Registers the ModelGenerator that owns this ModelConverter. """
        self._model_generator = model_generator

    @abstractmethod
    def convert(self) -> None:
        """ Sets self._model as defined in subclasses.

        This method should be called after the set_config method has prepared
        the converter.
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

    _model_generator: "KerasModelGenerator"

    def convert(self) -> None:
        self._model = tf.lite.TFLiteConverter.from_keras_model(
            self._model_generator._model
        ).convert()


class TFLiteQuantConverter(ModelConverter):
    """ Converts the _model field of a KerasModelGenerator to a quantized
    TFLite model.
    """

    _model_generator: "KerasModelGenerator"
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
        self._model = converter.convert()


class XCoreConverter(ModelConverter):
    """ Converts the _model field of a KerasModelGenerator to an xcore.ai-optimized
    TFLite model.
    """

    _model_generator: "KerasModelGenerator"
    _num_threads: int

    def __init__(
        self,
        model_generator: "KerasModelGenerator",
        quant_converter: TFLiteQuantConverter,
    ) -> None:
        """ Registers the ModelGenerator that owns this ModelConverter. 
        
        A hook for the source model must be specified as this converter could
        potentially be used with (among others) quantized or larq converted
        models.
        """
        self._model_generator = model_generator
        self._quant_converter = quant_converter

    def set_config(self, num_threads: Optional[int] = None) -> None:
        self._num_threads = num_threads or 1

    def convert(self) -> None:
        model = XCOREModel.deserialize(self._quant_converter._model)
        optimize_for_xcore(model, num_threads=self._num_threads)
        self._model = model.serialize()
