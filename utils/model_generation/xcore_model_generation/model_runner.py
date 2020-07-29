# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod
import numpy as np  # type: ignore

from typing import Dict, Any, Iterator, NamedTuple, Type, Optional

from .utils import quantize
from .model_converter import ModelConverter
from .model_generator import (
    ModelGenerator,
    Configuration,
    IntegrationTestModelGenerator,
)

ModelReports = Dict[ModelConverter, Any]


class ModelOutputs(NamedTuple):
    reference: np.ndarray
    xcore: np.ndarray


class ModelRun(NamedTuple):
    model_generator: ModelGenerator
    reports: Optional[ModelReports]
    outputs: Optional[ModelOutputs]


class ModelRunner(ABC):
    """ Superclass for defining the behavior of batched model generation runs.

    A ModelRunner registers a generator_class property, which is used by
    the generate_runs method. The generator_class gets run with all
    configurations defined in its builtin_configs method.
    """

    _model_generator: ModelGenerator
    _reports: Optional[ModelReports]
    _outputs: Optional[ModelOutputs]

    def __init__(self, generator_class: Type[ModelGenerator]) -> None:
        self.generator_class = generator_class
        self._reports = None
        self._outputs = None

    def generate_runs(self) -> Iterator[ModelRun]:
        """ Runs the model generation configs and yields reports and outputs.

        The reports and outputs are set by _run_model_generator.
        """
        for cfg in self.generator_class.builtin_configs():
            self._model_generator = self.generator_class()
            self._model_generator.set_config(**cfg)
            self._run_model_generator()
            yield ModelRun(self._model_generator, self._reports, self._outputs)
        del self._model_generator
        self._reports = self._outputs = None

    @abstractmethod
    def _run_model_generator(self) -> None:
        """ Defines how self._model_generator should be run with a config.

        Optionally sets self._reports and self._outputs.
        """
        self._model_generator.build()


class IntegrationTestRunner(ModelRunner):
    _model_generator: IntegrationTestModelGenerator

    def _run_model_generator(self) -> None:
        super()._run_model_generator()
        model_generator = self._model_generator
        model_generator._model.summary()  # TODO: remove this

        for converter in model_generator._converters:
            converter.convert()

        for evaluator in model_generator._evaluators:
            evaluator.evaluate()

        self._outputs = ModelOutputs(
            model_generator.reference_evaluator.output_data_quant,
            model_generator.xcore_evaluator.output_data,
        )
