# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod
import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore

from typing import Callable, Union

from .model_converter import TFLiteModel
from .utils import apply_interpreter_to_examples


class ModelEvaluator(ABC):
    _input_data: "np.ndarray"
    _output_data: "np.ndarray"

    def __init__(
        self, input_data_hook: Callable[[], Union[tf.Tensor, "np.ndarray"]]
    ) -> None:
        self._input_data_hook = input_data_hook

    @abstractmethod
    def evaluate(self) -> None:
        raise NotImplementedError()


class TFLiteEvaluator(ModelEvaluator):
    def __init__(
        self,
        input_data_hook: Callable[[], Union[tf.Tensor, "np.ndarray"]],
        model_hook: Callable[[], TFLiteModel],
    ) -> None:
        super().__init__(input_data_hook)
        self._model_hook = model_hook

    def evaluate(self) -> None:
        interpreter = tf.lite.Interpreter(model_content=self._model_hook())
        self._input_data = np.array(self._input_data_hook())
        self._output_data = apply_interpreter_to_examples(interpreter, self._input_data)
