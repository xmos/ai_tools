# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import pytest
from typing import Tuple

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import Padding
from tflite2xcore.transformation_passes import TdnnDeepConv2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_conv2d,
    ModelBuilder,
)

# from .conftest import PARAMS

PARAMS = {
    "default": {
        "input_height": [9],
        "input_width": [7],
        "input_channels": [4],
        "kernel_height": [2],
        "kernel_width": [2],
        "non_matching_input_channels": [9],
        "output_channels": [4],
        "non_matching_output_channels": [9],
        "padding": list(Padding),
        "stride_h": [1],
        "stride_w": [1],
    }
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model() -> ModelBuilder:
    return build_conv2d


@pytest.fixture()
def trf_pass() -> TdnnDeepConv2DPass:
    return TdnnDeepConv2DPass()


@pytest.fixture()
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


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_tdnn_mutate(trf_pass: TdnnDeepConv2DPass, model: XCOREModel) -> None:
    # run replacement pass
    trf_pass.run(model)
    model.sanity_check()
    pass


if __name__ == "__main__":
    pytest.main()
