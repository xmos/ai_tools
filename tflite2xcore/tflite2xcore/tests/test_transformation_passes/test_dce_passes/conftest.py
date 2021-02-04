# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType

from tflite2xcore.tests.test_transformation_passes.model_builders import build_relu


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()  # type: ignore
def model() -> XCOREModel:
    return build_relu(input_shape=[2, 2, 4], tensor_type=TensorType.INT8)
