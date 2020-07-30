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


class ModelRunner(ABC):
    """ Superclass for defining the behavior of batched model generation runs.

    A ModelRunner registers a generator_class property, which is used by
    the generate_runs method. The generator_class gets run with all
    configurations defined in its builtin_configs method.
    """

    _model_generator: ModelGenerator
    reports: Optional[ModelReports]
    outputs: Optional[ModelOutputs]

    def __init__(self, generator_class: Type[ModelGenerator]) -> None:
        self.generator_class = generator_class
        self.reports = None
        self.outputs = None

    @abstractmethod
    def run(self, cfg: Configuration) -> None:
        """ Defines how self._model_generator should be run with a config.

        Optionally sets self._reports and self._outputs.
        """
        self._model_generator = self.generator_class()
        self._model_generator.set_config(**cfg)
        self._model_generator.build()


class IntegrationTestRunner(ModelRunner):
    _model_generator: IntegrationTestModelGenerator

    def run(self, cfg: Configuration):
        super().run(cfg)
        model_generator = self._model_generator
        model_generator._model.summary()  # TODO: remove this

        for converter in model_generator._converters:
            converter.convert()

        for evaluator in model_generator._evaluators:
            evaluator.evaluate()

        self.outputs = ModelOutputs(
            model_generator.reference_evaluator.output_data_quant,
            model_generator.xcore_evaluator.output_data,
        )
