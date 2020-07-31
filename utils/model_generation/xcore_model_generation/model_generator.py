# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod
import tensorflow as tf  # type: ignore

from tflite2xcore.utils import set_all_seeds  # type: ignore # TODO: fix this

from .model_converter import (
    ModelConverter,
    TFLiteQuantConverter,
    XCoreConverter,
)
from .model_runner import ModelRunner, IntegrationTestRunner
from .model_evaluator import ModelEvaluator, TFLiteQuantEvaluator, XCoreEvaluator

from typing import Optional, Tuple, List, Dict, Any


Configuration = Dict[str, Any]


class ModelGenerator(ABC):
    """ Superclass for defining parameterized automatic model generation.

    The main use case of this class is generation of the integration test
    models.
    """

    _model: Any
    _config: Configuration = {}
    run: ModelRunner

    def __init__(
        self,
        runner: ModelRunner,
        converters: Optional[List[ModelConverter]] = None,
        evaluators: Optional[List[ModelEvaluator]] = None,
    ) -> None:
        """ Registers the runner, converters and evaluators. """
        self.run = runner
        self._converters = converters or []
        self._evaluators = evaluators or []

    @abstractmethod
    def build(self) -> None:
        """ Sets the _model field as needed by the subclass.
        
        The configuration should be set using the set_config method before
        calling this.
        """
        raise NotImplementedError()

    @abstractmethod
    def _set_config(self, cfg: Configuration) -> None:
        """ Sets the relevant configuration parameters and returns the unused ones.
        
        Should check if the given configuration parameters are legal.
        This method operates on the config input argument in-place.
        Subclasses should implement this instead of the set_config method.
        """
        for converter in self._converters:
            converter._set_config(cfg)

    def set_config(self, **config: Any) -> None:
        """ Configures the model generator before the build method is run.
        
        Default values for missing configuration parameters are set.
        Subclasses should implement the _set_config method instead of this.
        """
        self._set_config(config)
        if config:
            raise ValueError(
                f"Unexpected configuration parameter(s): {', '.join(config.keys())}"
            )


class KerasModelGenerator(ModelGenerator):
    _model: tf.keras.Model

    def _prep_backend(self) -> None:
        tf.keras.backend.clear_session()
        set_all_seeds()

    @property
    def _input_shape(self) -> Tuple[int, ...]:
        return self._model.input_shape[1:]  # type:ignore  # pylint: disable=no-member

    @property
    def _output_shape(self) -> Tuple[int, ...]:
        return self._model.output_shape[1:]  # type:ignore  # pylint: disable=no-member


class IntegrationTestModelGenerator(KerasModelGenerator):
    _quant_converter: TFLiteQuantConverter
    _xcore_converter: XCoreConverter
    reference_evaluator: TFLiteQuantEvaluator
    xcore_evaluator: XCoreEvaluator
    run: IntegrationTestRunner

    def __init__(self) -> None:
        self._quant_converter = TFLiteQuantConverter(self)
        self._xcore_converter = XCoreConverter(self, self._quant_converter)
        self.xcore_evaluator = XCoreEvaluator(
            self._quant_converter._get_representative_data,
            lambda: self._xcore_converter._model,
        )
        self.reference_evaluator = TFLiteQuantEvaluator(
            lambda: self.xcore_evaluator.input_data_float,
            lambda: self._quant_converter._model,
            lambda: self.xcore_evaluator.input_quant,
            lambda: self.xcore_evaluator.output_quant,
        )

        super().__init__(
            runner=IntegrationTestRunner(self),
            converters=[self._quant_converter, self._xcore_converter],
            evaluators=[self.xcore_evaluator, self.reference_evaluator],
        )

