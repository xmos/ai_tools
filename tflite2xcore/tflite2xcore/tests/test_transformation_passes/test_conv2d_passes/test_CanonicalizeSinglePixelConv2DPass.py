# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest
import numpy as np
from copy import deepcopy
from typing import Tuple

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import Padding, BuiltinOpCodes
from tflite2xcore.transformation_passes import CanonicalizeSinglePixelConv2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_conv2d,
    ModelBuilder,
)

from .conftest import (
    PARAMS,
    _test_non_matching_params,
    _test_replace_mutate,
    test_matching_params,
    test_non_matching_tensors,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------


PARAMS = deepcopy(PARAMS)

for params in PARAMS.values():
    params.update(
        {
            "input_channels": params["input_channels"]
            + params["non_matching_input_channels"],
            "output_channels": params["input_channels"]
            + params["non_matching_output_channels"],
            "input_height": params["kernel_height"] + params["input_height"],
            "input_width": params["kernel_width"] + params["input_width"],
        }
    )

PARAMS["extended"].update(
    {"input_height_deviation": [1, 2, 4], "input_width_deviation": [1, 2, 5]}
)
PARAMS["default"].update(
    {"input_height_deviation": [1, 4], "input_width_deviation": [2, 5]}
)
PARAMS["smoke"].update({"input_height_deviation": [1], "input_width_deviation": [2]})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model() -> ModelBuilder:
    return build_conv2d


@pytest.fixture()
def trf_pass() -> CanonicalizeSinglePixelConv2DPass:
    return CanonicalizeSinglePixelConv2DPass()


@pytest.fixture()
def weight_shape(
    output_channels: int, input_size: Tuple[int, int], input_channels: int
) -> Tuple[int, int, int, int]:
    return [output_channels, *input_size, input_channels]


@pytest.fixture()
def model(weight_shape: Tuple[int, int, int, int]) -> XCOREModel:
    return build_conv2d(
        weight_shape=weight_shape,
        input_size=weight_shape[1:3],
        # padding and stride should not matter for this model
        # but usind this builder guarantees the 1x1 output
        padding=Padding.VALID,
        strides=(1, 1),
    )


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(trf_pass: CanonicalizeSinglePixelConv2DPass, model: XCOREModel) -> None:
    subgraph = model.subgraphs[0]
    old_op = subgraph.operators[0]

    old_input, old_weights, old_bias = old_op.inputs[:3]
    old_output = old_op.outputs[0]

    old_weight_shape = old_weights.shape
    old_weight_quantization = old_weights.quantization
    old_fused_activation = old_op.builtin_options["fused_activation_function"]

    _test_replace_mutate(trf_pass, model, BuiltinOpCodes.FULLY_CONNECTED)
    new_op = subgraph.operators[0]
    new_weights = new_op.inputs[1]

    # check tensor objects
    assert old_input == new_op.inputs[0]
    assert old_weights != new_weights
    assert old_bias == new_op.inputs[2]
    assert old_output == new_op.outputs[0]

    # check weight tensor
    new_weight_shape = new_weights.shape
    assert len(new_weight_shape) == 2
    assert old_weight_shape[0] == new_weight_shape[0]
    assert np.prod(old_weight_shape[1:]) == new_weight_shape[1]
    assert old_fused_activation == new_op.builtin_options["fused_activation_function"]
    assert old_weight_quantization == new_weights.quantization


def test_non_matching_input_size(
    trf_pass: CanonicalizeSinglePixelConv2DPass,
    build_model: ModelBuilder,
    weight_shape: Tuple[int, int, int, int],
    input_height_deviation: int,
    input_width_deviation: int,
) -> None:
    input_size = (
        weight_shape[1] + input_height_deviation,
        weight_shape[2] + input_width_deviation,
    )
    model = build_conv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        # valid padding and strides equal to the input size ensures (1, 1) output
        padding=Padding.VALID,
        strides=input_size,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_output_size(
    trf_pass: CanonicalizeSinglePixelConv2DPass,
    build_model: ModelBuilder,
    weight_shape: Tuple[int, int, int, int],
) -> None:
    model = build_conv2d(
        weight_shape=weight_shape,
        input_size=weight_shape[1:3],
        padding=Padding.SAME,  # this assumes that input is never 1x1
        strides=(1, 1),
    )
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
