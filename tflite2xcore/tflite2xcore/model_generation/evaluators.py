# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore
from abc import abstractmethod
from typing import Union

from tflite2xcore.interpreters import XCOREInterpreter, XCOREDeviceInterpreter  # type: ignore # TODO: fix this
from tflite2xcore.utils import quantize, QuantizationTuple  # type: ignore # TODO: fix this
from tflite2xcore.utils import apply_interpreter_to_examples

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

    def __init__(
        self,
        runner: Runner,
        input_data_hook: Hook[Union[tf.Tensor, np.ndarray]],
        model_hook: Hook[TFLiteModel],
    ) -> None:
        super().__init__(runner, input_data_hook, model_hook)

    def evaluate(self) -> None:
        interpreter = tf.lite.Interpreter(model_content=self._model_hook())
        interpreter.allocate_tensors()

        self.input_data = self._input_data_hook()
        self.output_data = apply_interpreter_to_examples(interpreter, self.input_data)


class TFLiteQuantEvaluator(TFLiteEvaluator):
    """ Defines the evaluation logic for a TFLite quant model.
    
    Since the quantizer leaves in a float interface, input/output
    quantization parameters are required when getting the quantized values.
    """

    def __init__(
        self,
        runner: Runner,
        input_data_hook: Hook[Union[tf.Tensor, np.ndarray]],
        model_hook: Hook[TFLiteModel],
    ) -> None:
        super().__init__(runner, input_data_hook, model_hook)

    def get_input_data_quant(self, quant: QuantizationTuple) -> np.ndarray:
        return quantize(self.input_data, *quant)

    def get_output_data_quant(self, quant: QuantizationTuple) -> np.ndarray:
        return quantize(self.output_data, *quant)


class XCoreEvaluator(TFLiteEvaluator):
    """ Defines the evaluation logic for a TFLite float model. 
    
    The input and output quantization parameters are inferred from the model.
    """

    _input_quant: QuantizationTuple
    _output_quant: QuantizationTuple

    def __init__(
        self,
        runner: Runner,
        input_data_hook: Hook[Union[tf.Tensor, np.ndarray]],
        model_hook: Hook[TFLiteModel],
        use_device: bool = False,
    ) -> None:
        super().__init__(runner, input_data_hook, model_hook)
        self._use_device = use_device

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

    def get_interpreter(self) -> XCOREInterpreter:
        if self._use_device:
            return XCOREDeviceInterpreter(model_content=self._model_hook())
        else:
            return XCOREInterpreter(model_content=self._model_hook())

    def evaluate(self) -> None:
        interpreter = self.get_interpreter()
        interpreter.allocate_tensors()

        self._input_quant = QuantizationTuple(
            *interpreter.get_input_details()[0]["quantization"]
        )
        self._output_quant = QuantizationTuple(
            *interpreter.get_output_details()[0]["quantization"]
        )

        self.input_data_float = np.array(self._input_data_hook())
        self.input_data = quantize(self.input_data_float, *self.input_quant)

        self.output_data = apply_interpreter_to_examples(interpreter, self.input_data)
