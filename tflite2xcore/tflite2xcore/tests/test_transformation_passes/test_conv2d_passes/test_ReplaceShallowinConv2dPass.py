# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
from typing import Tuple
from copy import deepcopy

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import XCOREOpCodes, Padding
from tflite2xcore.transformation_passes import ReplaceShallowinConv2dPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    ModelBuilder,
    build_conv2d,
)
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_output_channels,
    test_non_matching_input_channels,
    test_non_matching_tensors,
)
from .test_ReplaceDeepConv2dPass import test_mutate as _test_mutate


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {"input_channels": list(range(4, 36, 4)), "kernel_width": list(range(1, 9))}
)

PARAMS["default"].update({"input_channels": [4, 8, 16], "kernel_width": [2, 3, 5]})

PARAMS["smoke"].update({"input_channels": [4, 8], "kernel_width": [3, 5]})

for k in PARAMS:
    all_tails = [
        (kw, cin)
        for cin in PARAMS[k]["input_channels"]
        for kw in PARAMS[k]["kernel_width"]
    ]
    PARAMS[k].update(
        weight_tail=[t for t in all_tails if t[0] * t[1] <= 32],
        non_matching_weight_tail=[t for t in all_tails if t[0] * t[1] > 32],
    )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model() -> ModelBuilder:
    return build_conv2d


@pytest.fixture()
def trf_pass() -> ReplaceShallowinConv2dPass:
    return ReplaceShallowinConv2dPass()


@pytest.fixture()
def weight_shape(
    output_channels: int, kernel_height: int, weight_tail: int
) -> Tuple[int, int, int]:
    return [output_channels, kernel_height, *weight_tail]


@pytest.fixture()
def model(
    weight_shape: Tuple[int, int, int],
    input_size: Tuple[int, int],
    padding: Padding,
    strides: Tuple[int, int],
) -> XCOREModel:
    return build_conv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
    )


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(trf_pass: ReplaceShallowinConv2dPass, model: XCOREModel) -> None:
    subgraph = model.subgraphs[0]
    K_w = subgraph.operators[0].inputs[1].shape[2]

    _test_mutate(trf_pass, model, custom_opcode=XCOREOpCodes.XC_conv2d_shallowin)

    custom_options = subgraph.operators[-1].custom_options
    assert "Kw" in custom_options
    assert custom_options["Kw"] == K_w


def test_non_matching_weight_tail(
    trf_pass: ReplaceShallowinConv2dPass,
    build_model: ModelBuilder,
    output_channels: int,
    kernel_height: int,
    non_matching_weight_tail: Tuple[int, int],
    input_size: Tuple[int, int],
    padding: Padding,
    strides: Tuple[int, int],
) -> None:
    model = build_model(
        weight_shape=[output_channels, kernel_height, *non_matching_weight_tail],
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
