# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import TensorType
from tflite2xcore.tests.test_transformation_passes.model_builders import build_relu
from ..conftest import (
    _test_non_matching_params,
    test_matching_params
)


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------

def count_tensors(model):
    return sum(len(subgraph.tensors) for subgraph in model.subgraphs)


def add_dangling_tensor(model):
    model.subgraphs[0].create_tensor(
        'dangling_tensor', TensorType.INT16, shape=[1, 32, 1, 1]
    )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------

@pytest.fixture()
def model():
    return build_relu(input_shape=[2, 2, 4], tensor_type=TensorType.INT8)
