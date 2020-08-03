# Copyright (c) 2020, XMOS Ltd, All rights reserved

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Dict, Any, NamedTuple, Optional

if TYPE_CHECKING:
    import numpy as np  # type: ignore
    from .model_converter import ModelConverter
    from .model_generator import (
        ModelGenerator,
        Configuration,
        IntegrationTestModelGenerator,
    )


ModelReports = Dict["ModelConverter", Any]


class ModelOutputs(NamedTuple):
    reference: "np.ndarray"
    xcore: "np.ndarray"


class ModelRunner(ABC):
    """ Superclass for defining the behavior of model generation runs.

    A ModelRunner is registered when a ModelGenerator object is instantiated.
    """

    reports: Optional[ModelReports]
    outputs: Optional[ModelOutputs]

    def __init__(self, generator: "ModelGenerator") -> None:
        """ Registers the generator that owns the runner. """
        self._model_generator = generator
        self.reports = None
        self.outputs = None

    @abstractmethod
    def __call__(self) -> None:
        """ Defines how self._model_generator should be run with a config.

        Optionally sets self.reports and self.outputs.
        """
        self._model_generator.build()


class IntegrationTestRunner(ModelRunner):
    _model_generator: "IntegrationTestModelGenerator"

    def __call__(self) -> None:
        """ Defines how an IntegrationTestModelGenerator should be run.
        
        The integration tests require the 'reference' and 'xcore' fields of
        self.outputs to be set.
        """
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
