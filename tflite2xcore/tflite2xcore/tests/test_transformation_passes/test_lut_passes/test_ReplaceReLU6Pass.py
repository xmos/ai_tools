# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from tflite2xcore.xcore_schema import TensorType
from tflite2xcore.transformation_passes import ReplaceReLU6Pass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_relu6
from .conftest import (
    PARAMS,
    test_matching_params,
    test_non_matching_input_type,
    test_non_matching_output_type,
    test_mutate,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return ReplaceReLU6Pass()


@pytest.fixture()
def model(input_shape):
    return build_relu6(input_shape=input_shape, tensor_type=TensorType.INT8)


if __name__ == "__main__":
    pytest.main()
