# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
from typing import Tuple

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import XCOREOpCodes, Padding
from tflite2xcore.transformation_passes import (
    ModelTransformationPass,
    ReplaceDeepConv2dPass,
)

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_conv2d,
    ModelBuilder,
)
from .conftest import (
    PARAMS,
    test_replace_mutate as _test_mutate,
    test_matching_params,
    test_non_matching_output_channels,
    test_non_matching_input_channels,
    test_non_matching_tensors,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()  # type: ignore
def build_model() -> ModelBuilder:
    return build_conv2d


@pytest.fixture()  # type: ignore
def trf_pass() -> ReplaceDeepConv2dPass:
    return ReplaceDeepConv2dPass()


@pytest.fixture()  # type: ignore
def model(
    weight_shape: Tuple[int, int, int, int],
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


@pytest.fixture()  # type: ignore
def custom_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_conv2d_deep


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(
    trf_pass: ModelTransformationPass, model: XCOREModel, custom_opcode: XCOREOpCodes
) -> None:
    subgraph = model.subgraphs[0]
    old_op = subgraph.operators[0]
    strides = tuple(old_op.builtin_options[f"stride_{ax}"] for ax in ("h", "w"))

    _test_mutate(trf_pass, model, custom_opcode)

    custom_options = subgraph.operators[-1].custom_options
    assert "stride" in custom_options
    assert custom_options["stride"] == strides

    assert "pad" in custom_options
    assert len(custom_options["pad"]) == 2


if __name__ == "__main__":
    pytest.main()
