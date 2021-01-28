# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import logging
import tensorflow as tf
from pathlib import Path
from typing import Optional
from tflite2xcore.utils import LoggingContext  # type: ignore # TODO: fix this
from tflite2xcore.xcore_schema import (  # type: ignore # TODO: fix this
    XCOREModel,
    XCOREOpCodes,
    BuiltinOpCodes,
    OperatorCode,
    TensorType,
)
from tflite2xcore.model_generation.data_factories import TensorDataFactory

from . import IntegrationTestModelGenerator, BinarizedTestRunner

from . import (  # pylint: disable=unused-import
    test_idempotence,
    #test_output,  # TODO: enable this
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


class CIFAR10DataFactory(TensorDataFactory):
    def make_data(self, batch: Optional[int] = None) -> tf.Tensor:
        _, (test_images, _) = tf.keras.datasets.cifar10.load_data()
        return tf.cast(test_images - 128.0, tf.int8)[:batch]


#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


class CIFAR10BinarizedTestRunner(BinarizedTestRunner):
    def make_repr_data_factory(self) -> TensorDataFactory:
        return CIFAR10DataFactory(self, lambda: self._model_generator.input_shape)


RUNNER = CIFAR10BinarizedTestRunner


#  ----------------------------------------------------------------------------
#                                   CONFIGS
#  ----------------------------------------------------------------------------

CONFIGS = {
    "default": {0: {"skip_on_device": False}},
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def abs_output_tolerance() -> int:
    return 0


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_converted_model(xcore_model: XCOREModel) -> None:
    subgraph = xcore_model.subgraphs[0]

    # check tensors
    assert len(subgraph.tensors) == 23

    assert len(subgraph.inputs) == 1
    input_tensor = subgraph.inputs[0]
    assert input_tensor.type is TensorType.INT8
    input_shape = input_tensor.shape
    assert len(input_shape) == 4
    assert input_shape[0] == 1
    assert input_shape[3] == 3

    assert len(subgraph.outputs) == 1
    output_tensor = subgraph.outputs[0]
    assert output_tensor.type is TensorType.INT8
    assert output_tensor.shape == (1, 10)

    # check operators
    assert len(subgraph.operators) == 9

    # check only first op
    assert len(input_tensor.consumers) == 1
    assert input_tensor.consumers[0].operator_code.code is BuiltinOpCodes.PAD

    opcode_cnt = xcore_model.count_operator_codes()
    assert opcode_cnt[OperatorCode(XCOREOpCodes.XC_bsign_8)] == 1
    assert opcode_cnt[OperatorCode(XCOREOpCodes.XC_bconv2d_bin)] == 1
    assert opcode_cnt[OperatorCode(XCOREOpCodes.XC_bconv2d_int8)] == 1
    assert opcode_cnt[OperatorCode(BuiltinOpCodes.PAD)] == 1
    assert opcode_cnt[OperatorCode(XCOREOpCodes.XC_conv2d_shallowin)] == 1
    assert opcode_cnt[OperatorCode(XCOREOpCodes.XC_pad)] == 2
    assert opcode_cnt[OperatorCode(XCOREOpCodes.XC_fc)] == 1
    assert opcode_cnt[OperatorCode(BuiltinOpCodes.SOFTMAX, version=2)] == 1


if __name__ == "__main__":
    pytest.main()
