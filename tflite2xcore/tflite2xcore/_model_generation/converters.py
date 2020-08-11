# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.converter import optimize_for_xcore  # type: ignore # TODO: fix this

from . import TFLiteModel, Configuration
from .utils import quantize_converter, parse_init_config

if TYPE_CHECKING:
    from .model_generators import ModelGenerator, KerasModelGenerator


class Converter(ABC):
    """ Superclass for defining model conversion logic and storing converted models.

    Converter objects are registered when a ModelGenerator object is
    instantiated.
    """

    _model: TFLiteModel

    def __init__(self, model_generator: "ModelGenerator") -> None:
        """ Registers the ModelGenerator that owns this Converter. """
        self._model_generator = model_generator

    @abstractmethod
    def convert(self) -> None:
        """ Sets self._model as defined in subclasses.

        This method should be called after the set_config method has prepared
        the converter.
        """
        raise NotImplementedError()

    @property
    def _config(self) -> "Configuration":
        return self._model_generator._config

    @abstractmethod
    def _set_config(self, cfg: "Configuration") -> None:
        """ Sets the relevant configuration parameters and returns the unused ones.
        
        This method operates on the config input argument in-place.
        """
        pass


class TFLiteFloatConverter(Converter):
    """ Converts the _model field of a KerasModelGenerator to a floating point
    TFLite model.
    """

    _model_generator: "KerasModelGenerator"

    def convert(self) -> None:
        self._model = tf.lite.TFLiteConverter.from_keras_model(
            self._model_generator._model
        ).convert()


class TFLiteQuantConverter(Converter):
    """ Converts the _model field of a KerasModelGenerator to a quantized
    TFLite model.
    """

    _model_generator: "KerasModelGenerator"
    _data_len: int

    def _set_config(self, cfg: "Configuration") -> None:
        self._config["input_init"] = cfg.pop("input_init", ("RandomUniform", -1, 1))
        self._data_len = 10

    def _get_representative_data(self) -> tf.Tensor:
        """ Returns the data to be used for post training quantization. """
        init = parse_init_config(*self._config["input_init"])
        return init((self._data_len, *self._model_generator._input_shape))

    def convert(self) -> None:
        converter = tf.lite.TFLiteConverter.from_keras_model(
            self._model_generator._model
        )
        quantize_converter(
            converter, representative_data=self._get_representative_data(),
        )
        self._model = converter.convert()


class XCoreConverter(Converter):
    """ Converts the _model field of a KerasModelGenerator to an xcore.ai-optimized
    TFLite model.
    """

    _model_generator: "KerasModelGenerator"

    def __init__(
        self, model_generator: "KerasModelGenerator", source_converter: Converter
    ) -> None:
        """ Registers the ModelGenerator that owns this Converter. 
        
        The source converter must be specified as this converter could
        potentially be used with (among others) quantized or larq converted
        models.
        """
        self._model_generator = model_generator
        self._source_converter = source_converter

    def _set_config(self, cfg: "Configuration") -> None:
        self._config["num_threads"] = cfg.pop("num_threads", 1)

    def convert(self) -> None:
        model = XCOREModel.deserialize(self._source_converter._model)
        optimize_for_xcore(model, num_threads=self._config["num_threads"])
        self._model = model.serialize()
