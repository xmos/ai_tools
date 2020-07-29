# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod
import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore

from typing import Callable, Union

from tflite2xcore.xcore_interpreter import XCOREInterpreter  # type: ignore # TODO: fix this

from .model_converter import TFLiteModel
from .utils import apply_interpreter_to_examples, quantize, dequantize


class ModelEvaluator(ABC):
    input_data: np.ndarray
    output_data: np.ndarray

    def __init__(
        self, input_data_hook: Callable[[], Union[tf.Tensor, np.ndarray]]
    ) -> None:
        self._input_data_hook = input_data_hook

    @abstractmethod
    def evaluate(self) -> None:
        raise NotImplementedError()


class TFLiteEvaluator(ModelEvaluator):
    def __init__(
        self,
        input_data_hook: Callable[[], Union[tf.Tensor, np.ndarray]],
        model_hook: Callable[[], TFLiteModel],
    ) -> None:
        super().__init__(input_data_hook)
        self._model_hook = model_hook

    def evaluate(self) -> None:
        interpreter = tf.lite.Interpreter(model_content=self._model_hook())
        self.input_data = np.array(self._input_data_hook())
        self.output_data = apply_interpreter_to_examples(interpreter, self.input_data)


class XCoreEvaluator(TFLiteEvaluator):
    def evaluate(self) -> None:
        interpreter = XCOREInterpreter(model_content=self._model_hook())
        interpreter.allocate_tensors()

        in_details = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()[0]
        in_ind, self._input_quant = in_details["index"], in_details["quantization"]
        out_ind, self._output_quant = out_details["index"], out_details["quantization"]

        self.input_data = np.array(self._input_data_hook())
        if self.input_data.dtype is np.dtype(np.float32):
            self.input_data = quantize(self.input_data, *self._input_quant)

        self.output_data = apply_interpreter_to_examples(
            interpreter,
            self.input_data,
            interpreter_input_ind=in_ind,
            interpreter_output_ind=out_ind,
        )

    @property
    def input_data_dequantized(self) -> np.ndarray:
        return dequantize(self.input_data, *self._input_quant)

    @property
    def output_data_dequantized(self) -> np.ndarray:
        return dequantize(self.output_data, *self._output_quant)
