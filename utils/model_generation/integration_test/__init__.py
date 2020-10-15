# Copyright (c) 2020, XMOS Ltd, All rights reserved

import logging
import pytest  # type: ignore
import _pytest  # type: ignore # NOTE: for typing only
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from abc import abstractmethod
from pathlib import Path
from typing import (
    Union,
    List,
    NamedTuple,
    Tuple,
    Dict,
    Optional,
    Iterable,
    Type,
    Optional,
)

from tflite2xcore.utils import dequantize  # type: ignore # TODO: fix this
from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.model_generation import (
    TFLiteModel,
    ModelGenerator,
)
from tflite2xcore.model_generation.runners import Runner
from tflite2xcore.model_generation.evaluators import (
    TFLiteEvaluator,
    TFLiteQuantEvaluator,
    XCoreEvaluator,
)
from tflite2xcore.model_generation.converters import (
    TFLiteFloatConverter,
    TFLiteQuantConverter,
    XCoreConverter,
)
from tflite2xcore.model_generation.data_factories import InputInitializerDataFactory
from tflite2xcore.interpreters.exceptions import (
    ModelSizeError,
    ArenaSizeError,
)


#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


class IntegrationTestRunner(Runner):
    _model_generator: "IntegrationTestModelGenerator"

    def __init__(
        self,
        generator: Type["IntegrationTestModelGenerator"],
        *,
        use_device: bool = False,
    ) -> None:
        super().__init__(generator)
        self._use_device = use_device

        self._xcore_converter = XCoreConverter(self, self.get_xcore_reference_model)
        self.register_converter(self._xcore_converter)

        self._identity_converter = XCoreConverter(
            self, self._xcore_converter.get_converted_model
        )
        self.register_converter(self._identity_converter)

        self._xcore_evaluator = XCoreEvaluator(
            self,
            self.get_xcore_evaluation_data,
            self._xcore_converter.get_converted_model,
            use_device=self._use_device,
        )
        self.register_evaluator(self._xcore_evaluator)

    @abstractmethod
    def get_xcore_reference_model(self) -> TFLiteModel:
        raise NotImplementedError()

    @abstractmethod
    def get_xcore_evaluation_data(self) -> TFLiteModel:
        raise NotImplementedError()

    @classmethod
    def load(cls, dirpath: Union[Path, str]) -> "IntegrationTestRunner":
        runner = super().load(dirpath)
        assert isinstance(runner, IntegrationTestRunner)
        return runner

    @abstractmethod
    def rerun_post_cache(self) -> None:
        raise NotImplementedError()


class DefaultIntegrationTestRunner(IntegrationTestRunner):
    class OutputData(NamedTuple):
        reference_float: np.ndarray
        reference_quant: np.ndarray
        xcore: np.ndarray

    _quantization_data: tf.Tensor
    outputs: "DefaultIntegrationTestRunner.OutputData"

    def __init__(
        self,
        generator: Type["IntegrationTestModelGenerator"],
        *,
        use_device: bool = False,
    ) -> None:
        super().__init__(generator, use_device=use_device)
        # quantization and test data
        self._repr_data_factory = InputInitializerDataFactory(
            self, lambda: self._model_generator.input_shape
        )
        self.register_data_factory(self._repr_data_factory)

        # floating point reference
        self._reference_float_converter = TFLiteFloatConverter(
            self, self.get_built_model
        )
        self.register_converter(self._reference_float_converter)

        self._reference_float_evaluator = TFLiteEvaluator(
            self,
            self.get_quantization_data,
            self._reference_float_converter.get_converted_model,
        )
        self.register_evaluator(self._reference_float_evaluator)

        # quantized reference
        self._reference_quant_converter = TFLiteQuantConverter(
            self, self.get_built_model, self.get_quantization_data
        )
        self.register_converter(self._reference_quant_converter)

        self._reference_quant_evaluator = TFLiteQuantEvaluator(
            self,
            self.get_quantization_data,
            self._reference_quant_converter.get_converted_model,
        )
        self.register_evaluator(self._reference_quant_evaluator)

    def get_xcore_reference_model(self) -> TFLiteModel:
        return self._reference_quant_converter.get_converted_model()

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

    def get_xcore_evaluation_data(self) -> TFLiteModel:
        return self.get_quantization_data()

    def run(self) -> None:
        """ Defines how a DefaultIntegrationTestRunner should be run.
        
        Most integration tests require self.outputs to be set.
        """
        super().run()
        self._reference_float_converter.convert()
        self._reference_quant_converter.convert()

        self._reference_quant_evaluator.evaluate()
        self._reference_float_evaluator.evaluate()

        self.rerun_post_cache()

    def rerun_post_cache(self) -> None:
        self._xcore_converter.convert()

        try:
            self._xcore_evaluator.evaluate()
        except ModelSizeError as e:
            if self._use_device and e.args[0].startswith("model_content too large: "):
                pytest.skip("Skipping due to excessive model size")
            else:
                raise
        except ArenaSizeError as e:
            if self._use_device:
                pytest.skip("Skipping (probably) due to excessive tensor arena size")
            else:
                raise

        self.outputs = self.OutputData(
            self._reference_float_evaluator.output_data,
            self._reference_quant_evaluator.get_output_data_quant(
                self._xcore_evaluator.output_quant
            ),
            self._xcore_evaluator.output_data,
        )
        self.converted_models.update(
            {
                "reference_quant": self._reference_quant_converter._model,
                "xcore": self._xcore_converter._model,
            }
        )

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

        self.dump_data(
            dirpath,
            data={
                "input": self._xcore_evaluator.input_data,
                "reference_quant_output": self.outputs.reference_quant,
                "xcore_output": self.outputs.xcore,
            },
            example_idx=example_idx,
        )


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
        msg = "Total" if ex_idx is None else f"Example {ex_idx}"
        if np.issubdtype(diff.dtype, np.integer):
            devs = [
                f"{c}/{diff.size} ({c / diff.size:.2%}) with diff={v}"
                for v, c in zip(*np.unique(diff, return_counts=True))
                if v
            ]
            msg += " deviations: " + (", ".join(devs) if devs else "None")
        else:
            stats = {
                "mean": np.mean(diff),
                "stdev": np.std(diff),
                "median": np.median(diff),
                "min": np.min(diff),
                "max": np.max(diff),
            }
            msg += f" deviation stats: {stats}"

        logger.log(level, msg)


def _compare_batched_arrays(
    predicted: np.ndarray, expected: np.ndarray, tolerance: Union[int, float]
) -> Dict[int, List[FailedElement]]:
    assert tolerance >= 0
    assert predicted.shape == expected.shape

    output_type = predicted.dtype
    assert output_type == expected.dtype  # NOTE: 'is' operator can be buggy, use ==
    if np.issubdtype(output_type, np.integer):
        diffs = np.int64(predicted) - np.int64(expected)
    elif np.issubdtype(output_type, np.floating):
        tolerance = np.float32(tolerance)
        diffs = np.float32(predicted) - np.float32(expected)
    else:
        raise TypeError("Only integer and float types are supported")

    failures: Dict[int, List[FailedElement]] = {}
    for j, (arr, arr_ref, diff) in enumerate(zip(predicted, expected, diffs)):
        __log_deviations(diff, logging.DEBUG, ex_idx=j)

        diff_idx = zip(*np.where(np.abs(diff) > tolerance))
        failed_examples = [
            FailedElement(idx, diff[idx], arr_ref[idx], arr[idx]) for idx in diff_idx
        ]
        if failed_examples:
            failures[j] = failed_examples
    __log_deviations(diffs, logging.INFO)
    return failures


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_output(
    run: DefaultIntegrationTestRunner,
    output_tolerance: Optional[Union[int, float]],
    request: _pytest.fixtures.SubRequest,
) -> None:
    if output_tolerance is None:
        # use implicitly derived tolerance
        output_quantization = run._xcore_evaluator.output_quant
        y_quant = run.outputs.reference_quant
        y_float = run.outputs.reference_float

        # The implicit tolerance is derived from how much the quantized reference
        # deviates from the floating point reference.
        max_diff = np.max(np.abs(dequantize(y_quant, *output_quantization) - y_float))
        # max_diff is usually at least 1 bit, but we ensure this and add some room for error
        output_tolerance = max(float(max_diff), output_quantization.scale) * 1.05
        logging.info(f"Using implicit output tolerance: {output_tolerance}")

        failures = _compare_batched_arrays(
            dequantize(run.outputs.xcore, *output_quantization),
            run.outputs.reference_float,
            output_tolerance,
        )
    else:
        failures = _compare_batched_arrays(
            run.outputs.xcore, run.outputs.reference_quant, output_tolerance
        )

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


def test_idempotence(xcore_model: XCOREModel, run: IntegrationTestRunner) -> None:
    run._identity_converter.convert()
    identical_model = XCOREModel.deserialize(run._identity_converter._model)
    assert xcore_model.is_equal(identical_model)
