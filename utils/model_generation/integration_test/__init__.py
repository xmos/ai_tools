# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import pytest  # type: ignore
import _pytest  # type: ignore # NOTE: for typing only
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from abc import abstractmethod
from pathlib import Path
from typing import Union, List, NamedTuple, Tuple, Dict, Optional, Iterable, Type

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.model_generation import (
    TFLiteModel,
    ModelGenerator,
)
from tflite2xcore.model_generation.runners import Runner
from tflite2xcore.model_generation.evaluators import (
    TFLiteQuantEvaluator,
    XCoreEvaluator,
)
from tflite2xcore.model_generation.converters import (
    TFLiteQuantConverter,
    XCoreConverter,
)
from tflite2xcore.model_generation.data_factories import InputInitializerDataFactory


#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


class IntegrationTestOutputData(NamedTuple):
    reference: np.ndarray
    xcore: np.ndarray


class IntegrationTestRunner(Runner):
    _model_generator: "IntegrationTestModelGenerator"
    _quantization_data: tf.Tensor
    outputs: IntegrationTestOutputData

    def __init__(self, generator: Type["IntegrationTestModelGenerator"]) -> None:
        self._repr_data_factory = InputInitializerDataFactory(
            self, lambda: self._model_generator.input_shape
        )

        self._reference_converter = TFLiteQuantConverter(
            self, lambda: self._model_generator._model, self.get_quantization_data
        )
        self._reference_evaluator = TFLiteQuantEvaluator(
            self, self.get_quantization_data, lambda: self._reference_converter._model
        )

        self._xcore_converter = XCoreConverter(
            self, lambda: self._reference_converter._model
        )
        self._identity_converter = XCoreConverter(
            self, lambda: self._xcore_converter._model
        )
        self._xcore_evaluator = XCoreEvaluator(
            self, self.get_quantization_data, lambda: self._xcore_converter._model
        )
        super().__init__(
            generator,
            converters=[
                self._reference_converter,
                self._xcore_converter,
                self._identity_converter,
            ],
            evaluators=[self._reference_evaluator, self._xcore_evaluator],
            data_factories=[self._repr_data_factory],
        )

    def get_quantization_data(self) -> tf.Tensor:
        try:
            return self._quantization_data
        except AttributeError:
            try:
                self._quantization_data = self._repr_data_factory.make_data(10)
            except AttributeError:
                raise Exception(
                    "Cannot get quantization data before runner is run!"
                ) from None
            return self._quantization_data

    def run(self) -> None:
        """ Defines how an IntegrationTestRunner should be run.
        
        The integration tests require the 'reference' and 'xcore' fields of
        self.outputs to be set.
        """
        super().run()
        self._converters[0].convert()
        self._evaluators[0].evaluate()
        self.rerun_post_cache()

    def rerun_post_cache(self) -> None:
        for converter in self._converters[1:]:
            converter.convert()

        for evaluator in self._evaluators[1:]:
            evaluator.evaluate()

        self.outputs = IntegrationTestOutputData(
            self._reference_evaluator.get_output_data_quant(
                self._xcore_evaluator.output_quant
            ),
            self._xcore_evaluator.output_data,
        )
        self.converted_models.update(
            {
                "reference": self._reference_converter._model,
                "xcore": self._xcore_converter._model,
                "xcore_identical": self._identity_converter._model,
            }
        )

    @classmethod
    def load(cls, dirpath: Union[Path, str]) -> "IntegrationTestRunner":
        runner = super().load(dirpath)
        assert isinstance(runner, IntegrationTestRunner)
        return runner

    def dump(
        self,
        dirpath: Path,
        example_idx: Union[int, Iterable[int]] = [],
        *,
        dump_models: bool = True,
        dump_visualizations: bool = True,
    ) -> None:
        if dump_models:
            self.dump_models(dirpath, visualize=dump_visualizations)

        data = {
            "input": self.get_quantization_data(),
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


class IntegrationTestModelGenerator(ModelGenerator):
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
    run_outputs: IntegrationTestOutputData,
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
