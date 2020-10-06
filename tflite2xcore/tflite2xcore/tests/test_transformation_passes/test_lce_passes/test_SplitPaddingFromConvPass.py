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

for params in PARAMS.values():
    params.update({"padding": [Padding.SAME]})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return SplitPaddingFromConvPass()


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    model = build_lceBconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    return model


def test_mutate(trf_pass, model):

    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 1

    original_input, original_output = subgraph.inputs[0], subgraph.outputs[0]

    options = subgraph.operators[0].custom_options
    strides = (options["stride_height"], options["stride_width"])

    assert len(subgraph.operators[0].inputs) == 3
    assert subgraph.operators[0].custom_options["padding"] is Padding.SAME

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

    assert conv_op.custom_options["padding"] is Padding.VALID

    # Check spacial dims of PAD output tensor is as expected
    paddings = pad_op.inputs[1].as_array()

    pad_output_shape = pad_op.outputs[0].shape

    for i in range(0, 4):
        assert (
            pad_op.inputs[0].shape[i] + sum(paddings[i]) == pad_output_shape[i]
        ), "bad output shape at index: " + str(i)


def test_non_matching_paddings(trf_pass, weight_shape, input_size, strides):

    model = build_lceBconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=Padding.VALID,
        strides=strides,
    )

    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
