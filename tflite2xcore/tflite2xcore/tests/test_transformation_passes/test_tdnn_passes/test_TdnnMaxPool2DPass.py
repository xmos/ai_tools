# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import pytest
from typing import Tuple


from tflite2xcore.transformation_passes.tdnn_passes import TdnnMaxPool2DPass
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import Padding, ActivationFunctionType

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_maxpool,
    ModelBuilder,
)

PARAMS = {
    "default": {
        "input_height": [1],
        "input_width": [9],
        "input_channels": [4],
        "padding": [Padding.VALID],
        "stride_h": [1],
        "stride_w": [1],
        "pool_h": [2],
        "pool_w": [2],
        "fused_activation": [ActivationFunctionType.NONE],
    }
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model() -> ModelBuilder:
    return build_maxpool


@pytest.fixture()
def trf_pass() -> TdnnMaxPool2DPass:
    return TdnnMaxPool2DPass()


@pytest.fixture()
def pool_size(pool_h, pool_w) -> Tuple[int, int]:
    return (pool_h, pool_w)


@pytest.fixture()
def model(
    build_model: ModelBuilder,
    input_shape: Tuple[int, int, int],
    pool_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: Padding,
    fused_activation: ActivationFunctionType,
) -> XCOREModel:
    return build_model(
        input_shape=input_shape,
        padding=padding,
        pool_size=pool_size,
        strides=strides,
        fused_activation=fused_activation,
    )


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_tdnn_mutate(trf_pass: TdnnMaxPool2DPass, model: XCOREModel) -> None:
    # run replacement pass
    trf_pass.run(model)
    model.sanity_check()

    # check operators
    subgraph = model.subgraphs[0]
    operators = subgraph.operators
    assert len(operators) == 2

    # check tensors
    op = operators[0] # pooling op
    assert len(op.inputs) == 1
    assert len(op.outputs) == 1

    op = operators[1] # ring buffer op
    assert len(op.inputs) == 2
    assert len(op.outputs) == 1

    # check wiring
    assert len(subgraph.get_tensor("input").consumers) == 1


if __name__ == "__main__":
    pytest.main()
