# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

import numpy as np

from copy import deepcopy

from tflite2xcore.transformation_passes import CanonicalizeConv2DInputChannels
from tflite2xcore.xcore_schema import BuiltinOpCodes

from tflite2xcore.tests.test_transformation_passes.model_builders import build_conv2d
from tflite2xcore.tests.test_transformation_passes.test_conv2d_passes.conftest import (
    PARAMS as CONV_PARAMS,
    test_non_matching_input_channels,
)
from .conftest import (
    _test_non_matching_params,
    test_matching_params,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(CONV_PARAMS)

for k in PARAMS:
    PARAMS[k]["input_channels"] = deepcopy(
        CONV_PARAMS[k]["non_matching_input_channels"]
    )
    PARAMS[k]["non_matching_input_channels"] = deepcopy(
        CONV_PARAMS[k]["input_channels"]
    )
    PARAMS[k]["output_channels"] = deepcopy(
        CONV_PARAMS["smoke"]["output_channels"]
        + CONV_PARAMS["smoke"]["non_matching_output_channels"]
    )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return CanonicalizeConv2DInputChannels()


@pytest.fixture()
def build_model():
    return build_conv2d


@pytest.fixture()
def weight_shape(output_channels, kernel_height, kernel_width, input_channels):
    return [output_channels, kernel_height, kernel_width, input_channels]


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    model = build_conv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
    )
    return model


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(trf_pass, model):
    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 1
    old_conv_op = subgraph.operators[0]
    old_weight_shape = old_conv_op.inputs[1].shape
    old_weights = old_conv_op.inputs[1].as_array()
    assert old_weights.dtype is np.dtype(np.int8)

    # run padding pass
    trf_pass.run(model)
    model.sanity_check()
    assert len(subgraph.operators) == 2

    # test pad operator
    pad_op = subgraph.operators[0]
    assert pad_op.operator_code.code is BuiltinOpCodes.PAD
    assert len(pad_op.inputs) == 2
    assert len(pad_op.outputs) == 1
    assert pad_op.inputs[0] in subgraph.inputs

    # test conv operator
    conv_op = subgraph.operators[1]
    assert conv_op.operator_code.code is BuiltinOpCodes.CONV_2D
    assert len(conv_op.inputs) == 3
    assert len(conv_op.outputs) == 1
    assert conv_op.outputs[0] in subgraph.outputs
    assert conv_op.inputs[0] is pad_op.outputs[0]

    # get channel counts
    input_channels = pad_op.inputs[0].shape[3]
    padded_channels = conv_op.inputs[0].shape[3]
    pad_size = padded_channels - input_channels

    # test weight tensor shape
    new_weight_shape = conv_op.inputs[1].shape
    assert old_weight_shape[:3] == new_weight_shape[:3]
    assert new_weight_shape[3] == padded_channels
    new_weights = conv_op.inputs[1].as_array()
    assert old_weights.dtype is np.dtype(np.int8)
    assert np.all(new_weights[..., :input_channels] == old_weights)
    assert np.all(
        new_weights[..., input_channels:]
        == np.zeros([*old_weight_shape[:3], pad_size], dtype=np.int8)
    )

    # test paddings tensor
    paddings = pad_op.inputs[1]
    pads_arr = paddings.as_array()
    assert pads_arr.shape == paddings.shape == (4, 2)
    assert pads_arr[0][0] == [0]
    assert pads_arr[0][1] == [0]
    assert pads_arr[1][0] == [0]
    assert pads_arr[1][1] == [0]
    assert pads_arr[2][0] == [0]
    assert pads_arr[2][1] == [0]
    assert pads_arr[3][0] == [0]
    assert pads_arr[3][1] == [pad_size]


if __name__ == "__main__":
    pytest.main()
