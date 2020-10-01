# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
from abc import abstractmethod
from typing import Union

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.converter import optimize_for_xcore  # type: ignore # TODO: fix this
from tflite2xcore.utils import quantize_converter  # type: ignore # TODO: fix this

from . import TFLiteModel, Configuration, Hook

from .runners import Runner, RunnerDependent


class Converter(RunnerDependent):
    """ Superclass for defining model conversion logic and storing converted models.

    Converter objects are registered in Runner objects.
    """

    _model: TFLiteModel

    def __init__(
        self,
        runner: Runner,
        input_model_hook: Hook[Union[TFLiteModel, tf.keras.Model]],
    ) -> None:
        self._runner = runner
        self._input_model_hook = input_model_hook

    @property
    def converted_model(self) -> TFLiteModel:
        try:
            return self._model
        except AttributeError:
            raise Exception(
                "Cannot get converted model before converter is run!"
            ) from None

    @abstractmethod
    def convert(self) -> None:
        """ Sets self._model as defined in subclasses.

        This method should be called after the set_config method has prepared
        the converter.
        """
        raise NotImplementedError()


class KerasModelConverter(Converter):
    """ Converts a Keras model to a TFLite model. """

    _input_model_hook: Hook[tf.keras.Model]

    def __init__(self, runner: Runner, input_model_hook: Hook[tf.keras.Model],) -> None:
        super().__init__(runner, input_model_hook)


class TFLiteFloatConverter(KerasModelConverter):
    """ Converts a Keras model to a floating point TFLite model. """

    def convert(self) -> None:
        self._model = tf.lite.TFLiteConverter.from_keras_model(
            self._input_model_hook()
        ).convert()


class TFLiteQuantConverter(KerasModelConverter):
    """ Converts a Keras model to a quantized TFLite model. """

    def __init__(
        self,
        runner: Runner,
        input_model_hook: Hook[tf.keras.Model],
        repr_data_hook: Hook[tf.Tensor],
    ) -> None:
        super().__init__(runner, input_model_hook)
        self._repr_data_hook = repr_data_hook

    def _set_config(self, cfg: Configuration) -> None:
        if "input_init" not in self._config:
            self._config["input_init"] = cfg.pop("input_init", ("RandomUniform", -1, 1))

    def convert(self) -> None:
        converter = tf.lite.TFLiteConverter.from_keras_model(self._input_model_hook())
        quantize_converter(converter, representative_data=self._repr_data_hook())
        self._model = converter.convert()


class XCoreConverter(Converter):
    """ Converts a (quantized) TFLite model to an xcore.ai-optimized TFLite model. """

    def __init__(self, runner: Runner, input_model_hook: Hook[TFLiteModel]) -> None:
        super().__init__(runner, input_model_hook)

    def _set_config(self, cfg: Configuration) -> None:
        if "num_threads" not in self._config:
            self._config["num_threads"] = cfg.pop("num_threads", 1)

    def convert(self) -> None:
        model = XCOREModel.deserialize(self._input_model_hook())
        optimize_for_xcore(model, num_threads=self._config["num_threads"])
        self._model = model.serialize()
