# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import os
import logging
import pathlib
import tempfile
import subprocess

import tensorflow as tf
import larq_compute_engine as lce
from abc import abstractmethod
from typing import Union

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.converter import optimize_for_xcore, XFORMER2_PATH
from tflite2xcore.utils import quantize_converter

from . import TFLiteModel, Configuration, Hook

from .runners import Runner, RunnerDependent

class Converter(RunnerDependent):
    """Superclass for defining model conversion logic and storing converted models.

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

    def get_converted_model(self) -> TFLiteModel:
        try:
            return self._model
        except AttributeError:
            raise Exception(
                "Cannot get converted model before converter is run!"
            ) from None

    @abstractmethod
    def convert(self) -> None:
        """Sets self._model as defined in subclasses.

        This method should be called after the set_config method has prepared
        the converter.
        """
        raise NotImplementedError()


class KerasModelConverter(Converter):
    """ Converts a Keras model to a TFLite model. """

    _input_model_hook: Hook[tf.keras.Model]

    def __init__(self, runner: Runner, input_model_hook: Hook[tf.keras.Model]) -> None:
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

    def convert(self) -> None:
        converter = tf.lite.TFLiteConverter.from_keras_model(self._input_model_hook())
        quantize_converter(converter, representative_data=self._repr_data_hook())
        self._model = converter.convert()


class XCoreConverter(Converter):
    """ Converts a (quantized) TFLite model to an xcore.ai-optimized TFLite model. """

    def __init__(
        self,
        runner: Runner,
        input_model_hook: Hook[TFLiteModel],
        *,
        experimental_xformer2: bool = False,
        only_experimental_xformer2: bool = False
    ) -> None:
        super().__init__(runner, input_model_hook)
        self._experimental_xformer2 = experimental_xformer2
        self._only_experimental_xformer2 = only_experimental_xformer2

    def _set_config(self, cfg: Configuration) -> None:
        if "num_threads" not in self._config:
            self._config["num_threads"] = cfg.pop("num_threads", 1)

    def convert(self) -> None:
        model = self._input_model_hook()
        if self._only_experimental_xformer2:
            with tempfile.TemporaryDirectory(suffix=str(os.getpid())) as dirname:
                input_path = pathlib.Path(dirname) / "input.tflite"

                with open(pathlib.Path(input_path).resolve(), "wb") as fd:
                    fd.write(model)

                from shutil import copyfile
                copyfile(input_path, "/Users/deepakpanickal/code/ai_tools/experimental/xformer/xformer_input.tflite")

                output_path = pathlib.Path(dirname) / "output.tflite"
                cmd = [str(XFORMER2_PATH), str(input_path), "-o", str(output_path)]
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
                logger = logging.getLogger()
                logger.debug(p.stdout)

                copyfile(output_path, "/Users/deepakpanickal/code/ai_tools/experimental/xformer/xformer_output.tflite")

                with open(pathlib.Path(output_path).resolve(), "rb") as fd:
                    bits = bytes(fd.read())
            self._model = bits

        else:
            model = XCOREModel.deserialize(model)
            model = optimize_for_xcore(
                model,
                num_threads=self._config["num_threads"],
                experimental_xformer2=self._experimental_xformer2,
            )
            self._model = model.serialize()


class LarqConverter(KerasModelConverter):
    """ Converts a Larq model to a TFLite model. """

    def convert(self) -> None:
        self._model = lce.convert_keras_model(
            self._input_model_hook(),
            inference_input_type=tf.int8,
            inference_output_type=tf.int8,
            target="xcore",
            experimental_enable_bitpacked_activations=True,
        )
