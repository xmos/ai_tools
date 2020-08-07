# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from copy import deepcopy

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes.lce_passes import SplitBsignPass
from tflite2xcore.xcore_schema import BuiltinOpCodes, TensorType

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
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------
@pytest.fixture()
def trf_pass():
    return SplitBsignPass()


@pytest.fixture()
def build_model():
    return build_lceBconv2d


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    model = build_lceBconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
        post_activation_bias=False,
        post_activation_mult=False,
    )
    return model


@pytest.fixture()
def model_with_float32_input(weight_shape, input_size, padding, strides):
    model = build_lceBconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
        post_activation_bias=False,
        post_activation_mult=False,
        input_tensor_type=TensorType.FLOAT32,
    )
    return model


@pytest.fixture()
def model_with_int32_input(weight_shape, input_size, padding, strides):
    model = build_lceBconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
        post_activation_bias=False,
        post_activation_mult=False,
        input_tensor_type=TensorType.INT32,
    )
    return model


def test_mutate(trf_pass, model):

    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 1

    in_ori, out_ori = subgraph.inputs[0], subgraph.outputs[0]

    assert (len(subgraph.operators[0].inputs)) == 2

    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    # need to clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    assert len(subgraph.operators) == 2
    op = subgraph.operators
    assert len(op[0].inputs) == 1
    assert len(op[0].outputs) == 1

    # check input/output tensors
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1

    assert in_ori is op[0].inputs[0]
    assert in_ori in subgraph.inputs

    assert all((i not in subgraph.inputs) for i in op[1].inputs)

    assert out_ori is op[1].outputs[0]
    assert out_ori in subgraph.outputs

    assert all((o not in subgraph.outputs) for o in op[0].outputs)


def test_non_matching_input_tensor_float32(trf_pass, model_with_float32_input):
    _test_non_matching_params(trf_pass, model_with_float32_input)


def test_non_matching_input_tensor_int32(trf_pass, model_with_int32_input):
    _test_non_matching_params(trf_pass, model_with_int32_input)


if __name__ == "__main__":
    pytest.main()
