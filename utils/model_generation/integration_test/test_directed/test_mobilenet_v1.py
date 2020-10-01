# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.applications import MobileNet  # type: ignore

from tflite2xcore.xcore_schema import (  # type: ignore # TODO: fix this
    XCOREOpCodes,
    BuiltinOpCodes,
    OperatorCode,
    TensorType,
)
from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore._model_generation import Configuration

from . import IntegrationTestModelGenerator, test_idempotence


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class MobileNetV1Model(IntegrationTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config["input_size"] = cfg.pop("input_size")
        super()._set_config(cfg)

    def _build_core_model(self) -> tf.keras.Model:
        input_size = self._config["input_size"]
        return MobileNet(input_shape=(input_size, input_size, 3))


GENERATOR = MobileNetV1Model

#  ----------------------------------------------------------------------------
#                                   CONFIGS
#  ----------------------------------------------------------------------------


CONFIGS = {
    "default": {0: {"input_size": 128}},
}


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_converted_model(xcore_model: XCOREModel) -> None:
    subgraph = xcore_model.subgraphs[0]

    # check tensors
    assert len(subgraph.tensors) == 90

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
    assert output_tensor.shape == (1, 1000)

    # check operators
    assert len(subgraph.operators) == 31

    # check only first op
    assert len(input_tensor.consumers) == 1
    assert input_tensor.consumers[0].operator_code.code is BuiltinOpCodes.PAD

    opcode_cnt = xcore_model.count_operator_codes()
    assert opcode_cnt[OperatorCode(XCOREOpCodes.XC_conv2d_1x1)] == 13
    assert opcode_cnt[OperatorCode(XCOREOpCodes.XC_conv2d_depthwise)] == 13
    assert opcode_cnt[OperatorCode(BuiltinOpCodes.PAD)] == 1
    assert opcode_cnt[OperatorCode(XCOREOpCodes.XC_conv2d_shallowin)] == 1
    assert opcode_cnt[OperatorCode(XCOREOpCodes.XC_avgpool2d_global)] == 1
    assert opcode_cnt[OperatorCode(XCOREOpCodes.XC_fc)] == 1
    assert opcode_cnt[OperatorCode(BuiltinOpCodes.SOFTMAX, version=2)] == 1


if __name__ == "__main__":
    pytest.main()
