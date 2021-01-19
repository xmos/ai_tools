# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf
import numpy as np
import larq_compute_engine as lce
from abc import abstractmethod
from typing import Union

from xcore_interpreters import XCOREInterpreter, XCOREDeviceInterpreter  # type: ignore # TODO: fix this

from tflite2xcore.utils import (
    quantize,
    QuantizationTuple,
    apply_interpreter_to_examples,
)

from . import TFLiteModel, Hook

from .runners import Runner, RunnerDependent


class Evaluator(RunnerDependent):
    """ Superclass for defining model evaluation logic.

    Evaluator objects are registered in Runner objects.
    Evaluation means that output data is generated for a given
    input, but it does not mean that a model is compared to another one.
    """

    _input_data: np.ndarray
    _output_data: np.ndarray

    def __init__(
        self,
        runner: Runner,
        input_data_hook: Hook[Union[tf.Tensor, np.ndarray]],
        model_hook: Hook[Union[tf.keras.Model, TFLiteModel]],
    ) -> None:
        self._runner = runner
        self._input_data_hook = input_data_hook
        self._model_hook = model_hook

    @property
    def input_data(self) -> np.ndarray:
        try:
            return self._input_data
        except AttributeError:
            raise Exception("Cannot get input data before evaluator is run!") from None

    @input_data.setter
    def input_data(self, data: Union[tf.Tensor, np.ndarray]) -> None:
        self._input_data = np.array(data)

    @property
    def output_data(self) -> np.ndarray:
        try:
            return self._output_data
        except AttributeError:
            raise Exception("Cannot get output data before evaluator is run!") from None

    @output_data.setter
    def output_data(self, data: Union[tf.Tensor, np.ndarray]) -> None:
        self._output_data = np.array(data)

    @abstractmethod
    def evaluate(self) -> None:
        """ Populates self._input_data and self._output_data. """
        raise NotImplementedError()


class TFLiteEvaluator(Evaluator):
    """ Defines the evaluation logic for a TFLite float model. """

    _interpreter: tf.lite.Interpreter

    def __init__(
        self,
        runner: Runner,
        input_data_hook: Hook[Union[tf.Tensor, np.ndarray]],
        model_hook: Hook[TFLiteModel],
    ) -> None:
        super().__init__(runner, input_data_hook, model_hook)

    def set_interpreter(self) -> None:
        self._interpreter = tf.lite.Interpreter(model_content=self._model_hook())

    def set_input_data(self) -> None:
        self.input_data = self._input_data_hook()

    def evaluate(self) -> None:
        self.set_interpreter()
        self._interpreter.allocate_tensors()
        self.set_input_data()
        self.output_data = apply_interpreter_to_examples(
            self._interpreter, self.input_data
        )
        del self._interpreter


class TFLiteQuantEvaluator(TFLiteEvaluator):
    """ Defines the evaluation logic for a quantized TFLite model. 

    The input and output quantization parameters are inferred from the model.
    """

    _input_type: np.dtype
    _input_quant: QuantizationTuple
    _output_quant: QuantizationTuple

    def __init__(
        self,
        runner: Runner,
        input_data_hook: Hook[Union[tf.Tensor, np.ndarray]],
        model_hook: Hook[TFLiteModel],
    ) -> None:
        super().__init__(runner, input_data_hook, model_hook)

    @property
    def input_quant(self) -> QuantizationTuple:
        try:
            return self._input_quant
        except AttributeError:
            raise Exception(
                "Cannot get input quantization before evaluator is run!"
            ) from None

    @property
    def output_quant(self) -> QuantizationTuple:
        try:
            return self._output_quant
        except AttributeError:
            raise Exception(
                "Cannot get output quantization before evaluator is run!"
            ) from None

    def set_input_data(self) -> None:
        input_details = self._interpreter.get_input_details()[0]
        self._input_quant = QuantizationTuple(*input_details["quantization"])
        self._input_type = np.dtype(input_details["dtype"])
        self._output_quant = QuantizationTuple(
            *self._interpreter.get_output_details()[0]["quantization"]
        )

        super().set_input_data()
        if (
            self._input_type in (np.int8, np.int16)
            and self.input_data.dtype == np.float32
        ):
            self.input_data = quantize(
                self.input_data, *self._input_quant, dtype=self._input_type
            )


class XCoreEvaluator(TFLiteQuantEvaluator):
    """ Defines the evaluation logic for a TFLite float model. 
    
    The input and output quantization parameters are inferred from the model.
    """

    def __init__(
        self,
        runner: Runner,
        input_data_hook: Hook[Union[tf.Tensor, np.ndarray]],
        model_hook: Hook[TFLiteModel],
        use_device: bool = False,
    ) -> None:
        super().__init__(runner, input_data_hook, model_hook)
        self._use_device = use_device

    def evaluate(self) -> None:
        if self._use_device:
            self._interpreter = XCOREDeviceInterpreter(model_content=self._model_hook())
        else:
            self._interpreter = XCOREInterpreter(model_content=self._model_hook())

        with self._interpreter:
            self._interpreter.allocate_tensors()
            self.set_input_data()
            self.output_data = apply_interpreter_to_examples(
                self._interpreter, self.input_data
            )

        del self._interpreter


class LarqEvaluator(Evaluator):
    def evaluate(self) -> None:
        interpreter = lce.tflite.python.interpreter.Interpreter(self._model_hook())
        self.input_data = self._input_data_hook()
        self.output_data = interpreter.predict(self.input_data)
