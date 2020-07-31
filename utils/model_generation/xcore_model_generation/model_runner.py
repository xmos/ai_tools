# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod
import numpy as np  # type: ignore

from typing import TYPE_CHECKING, Dict, Any, Iterator, NamedTuple, Type, Optional

from .utils import quantize
from .model_converter import ModelConverter

if TYPE_CHECKING:
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

    reports: Optional[ModelReports]
    outputs: Optional[ModelOutputs]

    def __init__(self, generator: "ModelGenerator") -> None:
        self._model_generator = generator
        self.reports = None
        self.outputs = None

    @abstractmethod
    def __call__(self) -> None:
        """ Defines how self._model_generator should be run with a config.

        Optionally sets self._reports and self._outputs.
        """
        self._model_generator.build()


class IntegrationTestRunner(ModelRunner):
    _model_generator: "IntegrationTestModelGenerator"

    def __call__(self) -> None:
        super().__call__()
        model_generator = self._model_generator

        for converter in model_generator._converters:
            converter.convert()

        for evaluator in model_generator._evaluators:
            evaluator.evaluate()

        self.outputs = ModelOutputs(
            model_generator.reference_evaluator.output_data_quant,
            model_generator.xcore_evaluator.output_data,
        )
