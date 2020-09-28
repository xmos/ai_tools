# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import pytest  # type: ignore
import _pytest  # type: ignore # NOTE: for typing only
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from abc import abstractmethod
from pathlib import Path
from typing import Union, List, NamedTuple, Tuple, Dict, Optional, Iterable

from tflite2xcore import tflite_visualize  # type: ignore # TODO: fix this
from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore._model_generation import TFLiteModel
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


class RunnerModels(NamedTuple):
    reference: TFLiteModel
    xcore: TFLiteModel
    xcore_identical: TFLiteModel


class IntegrationTestRunner(Runner):
    _model_generator: "IntegrationTestModelGenerator"
    outputs: RunnerOutputs
    models: RunnerModels

    def __call__(self) -> None:
        """ Defines how an IntegrationTestModelGenerator should be run.
        
        The integration tests require the 'reference' and 'xcore' fields of
        self.outputs to be set.
        """
        super().__call__()
        self._model_generator._reference_converter.convert()
        self._model_generator._reference_evaluator.evaluate()
        self.rerun_post_cache()

    def rerun_post_cache(self) -> None:
        model_generator = self._model_generator

        for converter in model_generator._converters[1:]:
            converter.convert()

        for evaluator in model_generator._evaluators[1:]:
            evaluator.evaluate()

        self.outputs = RunnerOutputs(
            model_generator._reference_evaluator.output_data_quant(
                model_generator._xcore_evaluator.output_quant
            ),
            model_generator._xcore_evaluator.output_data,
        )
        self.models = RunnerModels(
            model_generator._reference_converter._model,
            model_generator._xcore_converter._model,
            model_generator._identity_converter._model,
        )

    def dump(
        self,
        dirpath: Path,
        example_idx: Union[int, Iterable[int]] = [],
        dump_models: bool = True,
        dump_visualizations: bool = True,
    ) -> None:
        if dump_models:
            for name, model in self.models._asdict().items():
                name = "model_" + name
                model_ref_path = (dirpath / name).with_suffix(".tflite")
                model_ref_html = model_ref_path.with_suffix(".html")
                with open(model_ref_path, "wb") as f:
                    f.write(model)
                logging.debug(f"{name} dumped to {model_ref_path}")
                if dump_visualizations:
                    tflite_visualize.main(model_ref_path, model_ref_html)
                    logging.debug(f"{name} visualization dumped to {model_ref_html}")

        data = {
            "input": self._model_generator._xcore_evaluator.input_data,
            "reference_output": self.outputs.reference,
            "xcore_output": self.outputs.xcore,
        }
        example_idx = [example_idx] if isinstance(example_idx, int) else example_idx
        for key, arr in data.items():
            for j in example_idx:
                with open(dirpath / f"example_{j}.{key}", "wb") as f:
                    f.write(arr[j].flatten().tostring())


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class IntegrationTestModelGenerator(KerasModelGenerator):
    _reference_converter: TFLiteQuantConverter
    _xcore_converter: XCoreConverter
    _identity_converter: XCoreConverter
    _reference_evaluator: TFLiteQuantEvaluator
    _xcore_evaluator: XCoreEvaluator
    run: IntegrationTestRunner

    def __init__(self) -> None:
        self._reference_converter = TFLiteQuantConverter(self)
        self._reference_evaluator = TFLiteQuantEvaluator(
            self._reference_converter.get_representative_data,
            lambda: self._reference_converter._model,
        )

        self._xcore_converter = XCoreConverter(self, self._reference_converter)
        self._identity_converter = XCoreConverter(self, self._xcore_converter)
        self._xcore_evaluator = XCoreEvaluator(
            self._reference_converter.get_representative_data,
            lambda: self._xcore_converter._model,
        )

        self._identity_converter = XCoreConverter(self, self._xcore_converter)

        super().__init__(
            runner=IntegrationTestRunner(self),
            converters=[
                self._reference_converter,
                self._xcore_converter,
                self._identity_converter,
            ],
            evaluators=[self._reference_evaluator, self._xcore_evaluator],
        )

    @classmethod
    def load(cls, dirpath: Union[Path, str]) -> "IntegrationTestModelGenerator":
        gen = super().load(dirpath)
        assert isinstance(gen, IntegrationTestModelGenerator)
        return gen

    @abstractmethod
    def _build_core_model(self) -> tf.keras.Model:
        raise NotImplementedError()

    def build(self) -> None:
        self._prep_backend()
        self._model = self._build_core_model()
        self._model.build(self._model.input_shape)


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------


class FailedElement(NamedTuple):
    idx: Tuple[int, ...]
    diff: int
    expected: int
    predicted: int


def __log_deviations(
    diff: np.ndarray, level: int, *, ex_idx: Optional[int] = None
) -> None:
    logger = logging.getLogger()
    if logger.isEnabledFor(level):
        devs = [
            f"{c}/{diff.size} ({c / diff.size:.2%}) with diff={v}"
            for v, c in zip(*np.unique(diff, return_counts=True))
            if v
        ]
        msg = "Total" if ex_idx is None else f"Example {ex_idx}"
        msg += " deviations: " + (", ".join(devs) if devs else "None")
        logger.log(level, msg)


def _test_batched_arrays(
    predicted: np.ndarray, expected: np.ndarray, tolerance: Union[int, float]
) -> Dict[int, List[FailedElement]]:
    assert predicted.shape == expected.shape
    assert predicted.dtype is expected.dtype
    assert issubclass(predicted.dtype.type, np.integer)  # TODO: generalize to floats

    failures: Dict[int, List[FailedElement]] = {}
    diffs = np.int32(predicted) - np.int32(expected)
    for j, (arr, arr_ref, diff) in enumerate(zip(predicted, expected, diffs)):
        __log_deviations(diff, logging.DEBUG, ex_idx=j)

        diff_idx = zip(*np.where(np.abs(diff > tolerance)))
        failed_examples = [
            FailedElement(idx, diff[idx], arr_ref[idx], arr[idx]) for idx in diff_idx
        ]
        if failed_examples:
            failures[j] = failed_examples
    __log_deviations(diffs, logging.INFO)
    return failures


def _test_output(
    run_outputs: RunnerOutputs,
    request: _pytest.fixtures.SubRequest,
    tolerance: Union[int, float] = 1,
) -> None:
    failures = _test_batched_arrays(run_outputs.xcore, run_outputs.reference, tolerance)

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
            + ("" if verbose else "\nSet verbosity > 0 for more details."),
            pytrace=False,
        )


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_output(
    run: IntegrationTestRunner, request: _pytest.fixtures.SubRequest
) -> None:
    _test_output(run.outputs, request, tolerance=1)


def test_idempotence(
    xcore_model: XCOREModel, xcore_identical_model: XCOREModel
) -> None:
    assert xcore_model.is_equal(xcore_identical_model)
