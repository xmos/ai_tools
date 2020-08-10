# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod
import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore

from tflite2xcore.xcore_interpreter import XCOREInterpreter  # type: ignore # TODO: fix this

from .utils import apply_interpreter_to_examples, quantize, Quantization

from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    from .model_converter import TFLiteModel


class ModelEvaluator(ABC):
    """ Superclass for defining model evaluation logic.

    ModelEvaluator objects are registered when a ModelGenerator object is
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


class TFLiteEvaluator(ModelEvaluator):
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
        input_quant_hook: Callable[[], Quantization],
        output_quant_hook: Callable[[], Quantization],
    ) -> None:
        super().__init__(input_data_hook, model_hook)
        self._input_quant_hook = input_quant_hook
        self._output_quant_hook = output_quant_hook

    @property
    def input_data_quant(self) -> np.ndarray:
        return quantize(self.input_data, *self._input_quant_hook())

    @property
    def output_data_quant(self) -> np.ndarray:
        return quantize(self.output_data, *self._output_quant_hook())


class XCoreEvaluator(TFLiteEvaluator):
    """ Defines the evaluation logic for a TFLite float model. 
    
    The input and output quantization parameters are inferred from the model.
    """

    input_quant: Quantization
    output_quant: Quantization

    def evaluate(self) -> None:
        interpreter = XCOREInterpreter(model_content=self._model_hook())
        interpreter.allocate_tensors()

        self.input_quant = Quantization(
            *interpreter.get_input_details()[0]["quantization"]
        )
        self.output_quant = Quantization(
            *interpreter.get_output_details()[0]["quantization"]
        )

        self.input_data_float = np.array(self._input_data_hook())
        self.input_data = quantize(self.input_data_float, *self.input_quant)

        self.output_data = apply_interpreter_to_examples(interpreter, self.input_data)
