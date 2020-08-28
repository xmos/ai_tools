# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest

from copy import deepcopy

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes.lce_passes import ReplaceLceBconv2DPass
from tflite2xcore.xcore_schema import TensorType
from ..model_builders import build_lceBconv2d
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_output_channels,
    test_non_matching_input_channels,
    test_non_matching_stride_h,
    test_non_matching_stride_w,
    test_non_matching_dilation_w_factor,
    test_non_matching_dilation_h_factor,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------
@pytest.fixture()
def trf_pass():
    return ReplaceLceBconv2DPass(TensorType.INT8)


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    return build_lceBconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
        post_activation_mult=False,
        post_activation_bias=False,
        input_tensor_type=TensorType.INT8,
    )

#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------

def test_mutate(trf_pass, model):

    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 1

    trf_pass.run(model)
    model.sanity_check()

    CleanupManager(model).run_passes()
    model.sanity_check()
    
    assert len(subgraph.operators) == 1


if __name__ == "__main__":
    pytest.main()
