# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import logging
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Optional, Type, Any
from tflite2xcore.utils import LoggingContext
from tflite2xcore.model_generation.data_factories import TensorDataFactory

from . import IntegrationTestModelGenerator, BinarizedTestRunner

from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class BNNModelGenerator(IntegrationTestModelGenerator):
    def _build_core_model(self) -> tf.keras.Model:
        # tf may complain about missing gradients, so silence it
        with LoggingContext(tf.get_logger(), logging.ERROR):
            return tf.keras.models.load_model(
                Path(__file__).parent / "bnn_model", compile=False
            )


GENERATOR = BNNModelGenerator

#  ----------------------------------------------------------------------------
#                                DATA FACTORIES
#  ----------------------------------------------------------------------------


class CIFAR10TestDataFactory(TensorDataFactory):
    def make_data(self, batch: Optional[int] = None) -> tf.Tensor:
        _, (test_images, _) = tf.keras.datasets.cifar10.load_data()
        assert self._shape_hook() == test_images.shape[1:]
        return tf.cast(test_images - 128.0, tf.int8)[:batch]


class CIFAR10TestLabelFactory(TensorDataFactory):
    def make_data(self, batch: Optional[int] = None) -> tf.Tensor:
        _, (_, test_labels) = tf.keras.datasets.cifar10.load_data()
        return tf.cast(test_labels, tf.int8)[:batch, 0]


#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


class CIFAR10BinarizedTestRunner(BinarizedTestRunner):
    def __init__(
        self,
        generator: Type[IntegrationTestModelGenerator],
        **kwargs: Any,
    ) -> None:
        super().__init__(generator, **kwargs)

        self._ground_truth_data_factory = CIFAR10TestLabelFactory(self, lambda: tuple())
        self.register_data_factory(self._ground_truth_data_factory)

    @property
    def repr_data_example_count(self) -> int:
        # TODO: fix this when tools are more stable (on the device this tends to time out)
        return 100 if self._use_device else 10000

    def make_repr_data_factory(self) -> TensorDataFactory:
        return CIFAR10TestDataFactory(self, lambda: self._model_generator.input_shape)

    def get_ground_truth_data(self) -> tf.Tensor:
        return self._ground_truth_data_factory.make_data(self.repr_data_example_count)


RUNNER = CIFAR10BinarizedTestRunner


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture
def abs_output_tolerance(use_device: bool) -> int:
    return 13 if use_device else 49


@pytest.fixture
def expected_accuracy(use_device: bool) -> float:
    return 0.79 if use_device else 0.6882


@pytest.fixture
def expected_prediction_deviation(use_device: bool) -> int:
    return 0 if use_device else 84


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_prediction_deviation(
    run: CIFAR10BinarizedTestRunner, expected_prediction_deviation: int
) -> None:
    xcore_labels = np.argmax(run.outputs.xcore, axis=1)
    reference_labels = np.argmax(run.outputs.reference_quant, axis=1)
    deviation_indices = (reference_labels != xcore_labels).nonzero()[0]
    assert len(deviation_indices) == expected_prediction_deviation


def test_accuracy(run: CIFAR10BinarizedTestRunner, expected_accuracy: float) -> None:
    metric = tf.keras.metrics.Accuracy()
    metric.update_state(
        y_true=run.get_ground_truth_data(), y_pred=np.argmax(run.outputs.xcore, axis=1)
    )
    assert metric.result().numpy() == np.float32(expected_accuracy)


if __name__ == "__main__":
    pytest.main()
