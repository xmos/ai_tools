# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Union

from tflite2xcore.interpreters import XCOREInterpreter, XCOREDeviceInterpreter  # type: ignore # TODO: fix this
from tflite2xcore.utils import quantize, QuantizationTuple

from . import TFLiteModel
from .utils import apply_interpreter_to_examples


class Evaluator(ABC):
    """ Superclass for defining model evaluation logic.

    Evaluator objects are registered when a ModelGenerator object is
    instantiated. Evaluation means that output data is generated for a given
    input, but it does not mean that a model is compared to another one.
    """

    input_data: np.ndarray
    output_data: np.ndarray

    def __init__(
        self, input_data_hook: Callable[[], Union[tf.Tensor, np.ndarray]]
    ) -> None:
        """ Registers a callback that returns the input data. """
        self._input_data_hook = input_data_hook

    @abstractmethod
    def evaluate(self) -> None:
        """ Populates self.input_data and self.output_data. """
        raise NotImplementedError()


class TFLiteEvaluator(Evaluator):
    """ Defines the evaluation logic for a TFLite float model. """

    def __init__(
        self,
        input_data_hook: Callable[[], Union[tf.Tensor, np.ndarray]],
        model_hook: Callable[[], "TFLiteModel"],
    ) -> None:
        super().__init__(input_data_hook)
        self._model_hook = model_hook

    def evaluate(self) -> None:
        interpreter = tf.lite.Interpreter(model_content=self._model_hook())
        interpreter.allocate_tensors()

        self.input_data = np.array(self._input_data_hook())
        self.output_data = apply_interpreter_to_examples(interpreter, self.input_data)


class TFLiteQuantEvaluator(TFLiteEvaluator):
    """ Defines the evaluation logic for a TFLite quant model.
    
    Since the quantizer leaves in a float interface, callbacks for input/output
    quantization parameters are required.
    """

    def __init__(
        self,
        input_data_hook: Callable[[], Union[tf.Tensor, np.ndarray]],
        model_hook: Callable[[], "TFLiteModel"],
    ) -> None:
        super().__init__(input_data_hook, model_hook)

    def input_data_quant(self, quant: QuantizationTuple) -> np.ndarray:
        return quantize(self.input_data, *quant)

    def output_data_quant(self, quant: QuantizationTuple) -> np.ndarray:
        return quantize(self.output_data, *quant)


class XCoreEvaluator(TFLiteEvaluator):
    """ Defines the evaluation logic for a TFLite float model. 
    
    The input and output quantization parameters are inferred from the model.
    """

    input_quant: QuantizationTuple
    output_quant: QuantizationTuple

    def __init__(
        self,
        input_data_hook: Callable[[], Union[tf.Tensor, np.ndarray]],
        model_hook: Callable[[], "TFLiteModel"],
        use_device=False,
    ) -> None:
        super().__init__(input_data_hook, model_hook)
        self._use_device = use_device

    def get_interpreter(self):
        if self._use_device:
            return XCOREDeviceInterpreter(model_content=self._model_hook())
        else:
            return XCOREInterpreter(model_content=self._model_hook())

    def evaluate(self) -> None:
        interpreter = self.get_interpreter()
        interpreter.allocate_tensors()

        self.input_quant = QuantizationTuple(
            *interpreter.get_input_details()[0]["quantization"]
        )
        self.output_quant = QuantizationTuple(
            *interpreter.get_output_details()[0]["quantization"]
        )

        self.input_data_float = np.array(self._input_data_hook())
        self.input_data = quantize(self.input_data_float, *self.input_quant)

        self.output_data = apply_interpreter_to_examples(interpreter, self.input_data)
