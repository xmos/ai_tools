# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

import numpy as np

from copy import deepcopy

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes.lce_passes import SplitPaddingFromConvPass
from tflite2xcore.xcore_schema import BuiltinOpCodes, TensorType, Padding

from ..model_builders import build_lceBconv2d
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    # test_matching_params,
    # test_non_matching_output_channels,
    # test_non_matching_input_channels,
    # test_non_matching_stride_h,
    # test_non_matching_stride_w,
    # test_non_matching_dilation_w_factor,
    # test_non_matching_dilation_h_factor,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {"padding": [Padding.SAME],}
)

PARAMS["default"].update(
    {"padding": [Padding.SAME],}
)

PARAMS["smoke"].update(
    {"padding": [Padding.SAME],}
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------
@pytest.fixture()
def trf_pass():
    return SplitPaddingFromConvPass()


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
def model_with_non_matching_padding(weight_shape, input_size, padding, strides):
    model = build_lceBconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=Padding.VALID,
        strides=strides,
        post_activation_bias=False,
        post_activation_mult=False,
    )
    return model


def test_mutate(trf_pass, model):

    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 1

    original_input, original_output = subgraph.inputs[0], subgraph.outputs[0]

    options = subgraph.operators[0].custom_options
    strides = strides = (options["stride_height"], options["stride_width"])

    assert (len(subgraph.operators[0].inputs)) == 2
    assert subgraph.operators[0].custom_options["padding"] == 1

    # Run the pass
    trf_pass.run(model)
    model.sanity_check()

    # Clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    assert len(subgraph.operators) == 2

    pad_op = subgraph.operators[0]
    conv_op = subgraph.operators[1]

    assert len(pad_op.inputs) == 2
    assert len(pad_op.outputs) == 1

    # Check input/output tensors
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1

    assert original_input is pad_op.inputs[0]
    assert original_input in subgraph.inputs

    assert all((i not in subgraph.inputs) for i in conv_op.inputs)

    assert original_output is conv_op.outputs[0]
    assert original_output in subgraph.outputs

    assert all((o not in subgraph.outputs) for o in pad_op.outputs)

    assert conv_op.outputs[0].shape == original_output.shape

    assert conv_op.custom_options["padding"] == 2

    # Check spacial dims of PAD output tensor is as expected
    paddings = pad_op.inputs[1].as_array()
    out_height_expected = pad_op.inputs[0].shape[1] + paddings[1][0] + paddings[1][1]
    out_width_expected = pad_op.inputs[0].shape[2] + paddings[2][0] + paddings[2][1]

    pad_output_shape = pad_op.outputs[0].shape
    assert out_height_expected == pad_output_shape[1]
    assert out_width_expected == pad_output_shape[2]


def test_non_matching_paddings(trf_pass, model_with_non_matching_padding):
    _test_non_matching_params(trf_pass, model_with_non_matching_padding)


if __name__ == "__main__":
    pytest.main()
