# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_schema import TensorType
from tflite2xcore.transformation_passes import ReplaceTanhPass

from tflite2xcore.tests.test_transformation_passes.model_builders import build_tanh
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
    return ReplaceTanhPass()


@pytest.fixture()
def model(input_shape):
    return build_tanh(input_shape=input_shape, tensor_type=TensorType.INT8)


if __name__ == "__main__":
    pytest.main()
