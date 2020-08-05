# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from copy import deepcopy

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes.lce_passes import CanonicalizeLceBconv2DPass
from tflite2xcore.xcore_schema import BuiltinOpCodes

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
    test_non_matching_dilation_h_factor
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------
@pytest.fixture()
def trf_pass():
    return CanonicalizeLceBconv2DPass()

@pytest.fixture()
def build_model():
    return build_lceBconv2d

@pytest.fixture()
def model_without_post_act_mult_bias(weight_shape, input_size, padding, strides):
    return build_lceBconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
        post_activation_mult=False,
        post_activation_bias=False,
    )


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    model = build_lceBconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    return model

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------
def test_mutate(trf_pass, model):

    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 1

    in_ori, out_ori = subgraph.inputs[0], subgraph.outputs[0]

    assert (len(subgraph.operators[0].inputs)) == 4

    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    # need to clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    assert len(subgraph.operators) == 1
    op = subgraph.operators[0]
    assert len(op.inputs) == 2
    assert len(op.outputs) == 1

    # check input/output tensors
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1

    assert in_ori is op.inputs[0]
    assert in_ori in subgraph.inputs
    assert out_ori is op.outputs[0]
    assert out_ori in subgraph.outputs


def test_non_matching_wrong_input_tensor_count(
    trf_pass, model_without_post_act_mult_bias
):
    _test_non_matching_params(trf_pass, model_without_post_act_mult_bias)


if __name__ == "__main__":
    pytest.main()
