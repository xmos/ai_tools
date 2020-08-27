# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import pytest  # type: ignore
import _pytest  # type: ignore # NOTE: for typing only
import numpy as np  # type: ignore
from pathlib import Path
from typing import Union, List, NamedTuple, Tuple, Dict

from tflite2xcore import tflite_visualize  # type: ignore # TODO: fix this
from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
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


#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


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
            model_generator._reference_evaluator.output_data_quant,
            model_generator._xcore_evaluator.output_data,
        )

    @property
    def xcore_model(self) -> XCOREModel:
        return self._model_generator._xcore_converter._model


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class IntegrationTestModelGenerator(KerasModelGenerator):
    _reference_converter: TFLiteQuantConverter
    _xcore_converter: XCoreConverter
    _reference_evaluator: TFLiteQuantEvaluator
    _xcore_evaluator: XCoreEvaluator
    run: IntegrationTestRunner

    def __init__(self) -> None:
        self._reference_converter = TFLiteQuantConverter(self)
        self._xcore_converter = XCoreConverter(self, self._reference_converter)
        self._xcore_evaluator = XCoreEvaluator(
            self._reference_converter._get_representative_data,
            lambda: self._xcore_converter._model,
        )
        self._reference_evaluator = TFLiteQuantEvaluator(
            lambda: self._xcore_evaluator.input_data_float,
            lambda: self._reference_converter._model,
            lambda: self._xcore_evaluator.input_quant,
            lambda: self._xcore_evaluator.output_quant,
        )

        super().__init__(
            runner=IntegrationTestRunner(self),
            converters=[self._reference_converter, self._xcore_converter],
            evaluators=[self._xcore_evaluator, self._reference_evaluator],
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
                logging.debug(f"{name} dumped to {model_ref_path}")
                tflite_visualize.main(model_ref_path, model_ref_html)
                logging.debug(f"{name} visualization dumped to {model_ref_html}")
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
    predicted: np.ndarray, expected: np.ndarray, tolerance: Union[int, float]
) -> Dict[int, List[FailedElement]]:
    assert predicted.shape == expected.shape
    assert predicted.dtype is expected.dtype
    assert issubclass(predicted.dtype.type, np.integer)  # TODO: generalize to floats

    def collect_deviations(diff: np.ndarray) -> List[str]:
        return [
            f"{c}/{diff.size} ({c / diff.size:.2%}) with diff={v}"
            for v, c in zip(*np.unique(diff, return_counts=True))
            if v
        ]

    failures: Dict[int, List[FailedElement]] = {}
    diffs = np.abs(np.int32(predicted) - np.int32(expected))
    for j, (arr, arr_ref, diff) in enumerate(zip(predicted, expected, diffs)):
        devs = collect_deviations(diff)
        logging.debug(
            f"Example {j} deviations: " + (", ".join(devs) if devs else "None")
        )

        diff_idx = zip(*np.where(diff > tolerance))
        failed_examples = [
            FailedElement(idx, diff[idx], arr_ref[idx], arr[idx]) for idx in diff_idx
        ]
        if failed_examples:
            failures[j] = failed_examples
    devs = collect_deviations(diffs)
    logging.info(f"Total deviations: " + (", ".join(devs) if devs else "None"))
    return failures


def _test_output(
    run: IntegrationTestRunner,
    request: _pytest.fixtures.SubRequest,
    tolerance: Union[int, float] = 1,
) -> None:
    failures = _test_batched_arrays(run.outputs.xcore, run.outputs.reference, tolerance)

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


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_output(
    run: IntegrationTestRunner, request: _pytest.fixtures.SubRequest
) -> None:
    _test_output(run, request, tolerance=1)
