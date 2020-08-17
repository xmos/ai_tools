# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import _pytest  # type: ignore # NOTE: for typing only
import numpy as np  # type: ignore
from pathlib import Path
from typing import Union, List, NamedTuple, Tuple, Dict

from tflite2xcore import tflite_visualize  # type: ignore # TODO: fix this
from tflite2xcore import xlogging  # type: ignore # TODO: fix this
from tflite2xcore._model_generation.model_generators import KerasModelGenerator
from tflite2xcore._model_generation.runners import Runner, RunnerOutputs
from tflite2xcore._model_generation.evaluators import (
    TFLiteQuantEvaluator,
    XCoreEvaluator,
)
from tflite2xcore._model_generation.converters import (
    TFLiteQuantConverter,
    XCoreConverter,
)


class IntegrationTestRunner(Runner):
    _model_generator: "IntegrationTestModelGenerator"
    outputs: RunnerOutputs

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

        self.outputs = RunnerOutputs(
            model_generator.reference_evaluator.output_data_quant,
            model_generator.xcore_evaluator.output_data,
        )


class IntegrationTestModelGenerator(KerasModelGenerator):
    _reference_converter: TFLiteQuantConverter
    _xcore_converter: XCoreConverter
    reference_evaluator: TFLiteQuantEvaluator
    xcore_evaluator: XCoreEvaluator
    run: IntegrationTestRunner

    def __init__(self) -> None:
        self._reference_converter = TFLiteQuantConverter(self)
        self._xcore_converter = XCoreConverter(self, self._reference_converter)
        self.xcore_evaluator = XCoreEvaluator(
            self._reference_converter._get_representative_data,
            lambda: self._xcore_converter._model,
        )
        self.reference_evaluator = TFLiteQuantEvaluator(
            lambda: self.xcore_evaluator.input_data_float,
            lambda: self._reference_converter._model,
            lambda: self.xcore_evaluator.input_quant,
            lambda: self.xcore_evaluator.output_quant,
        )

        super().__init__(
            runner=IntegrationTestRunner(self),
            converters=[self._reference_converter, self._xcore_converter],
            evaluators=[self.xcore_evaluator, self.reference_evaluator],
        )

    def save(self, dirpath: Union[Path, str], dump_models: bool = False) -> Path:
        dirpath = super().save(dirpath)
        if dump_models:
            for name, model in [
                ("model_ref", self._reference_converter._model),
                ("model_xcore", self._xcore_converter._model),
            ]:
                model_ref_path = (dirpath / name).with_suffix(".tflite")
                model_ref_html = model_ref_path.with_suffix(".html")
                with open(model_ref_path, "wb") as f:
                    f.write(model)
                xlogging.logging.debug(f"{name} dumped to {model_ref_path}")
                tflite_visualize.main(model_ref_path, model_ref_html)
                xlogging.logging.debug(
                    f"{name} visualization dumped to {model_ref_html}"
                )
        return dirpath

    @classmethod
    def load(cls, dirpath: Union[Path, str]) -> "IntegrationTestModelGenerator":
        gen = super().load(dirpath)
        assert isinstance(gen, IntegrationTestModelGenerator)
        return gen


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------


class FailedElement(NamedTuple):
    idx: Tuple[int, ...]
    diff: int
    expected: int
    predicted: int


def _test_batched_arrays(
    predicted: np.ndarray, expected: np.ndarray, tolerance: Union[int, float] = 1
) -> Dict[int, List[FailedElement]]:
    assert predicted.shape == expected.shape
    assert predicted.dtype is expected.dtype

    failures: Dict[int, List[FailedElement]] = {}
    for j, (arr, arr_ref) in enumerate(zip(predicted, expected)):
        diff = np.abs(np.int32(arr) - np.int32(arr_ref))
        diff_idx = zip(*np.where(diff > tolerance))
        failures[j] = [
            FailedElement(idx, diff[idx], arr_ref[idx], arr[idx]) for idx in diff_idx
        ]
    return failures


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_output(
    run: IntegrationTestRunner, request: _pytest.fixtures.SubRequest
) -> None:
    failures = _test_batched_arrays(run.outputs.xcore, run.outputs.reference)

    verbose = request.config.getoption("verbose") > 0

    msg = "".join(
        f"\n{request.node.fspath}::{request.node.name} Example {j}"
        + (
            "".join(
                f"\nidx={e.idx}: diff={e.diff}, "
                f"expected={e.expected}, predicted={e.predicted}"
                for e in elements
            )
            if verbose
            else ""
        )
        for j, elements in failures.items()
    )

    if failures:
        pytest.fail(
            f"The following examples have failed elements: {msg}"
            + ("" if verbose else "\nSet verbsity > 0 for more details."),
            pytrace=False,
        )
