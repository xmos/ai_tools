# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes import FuseConv2dPaddingPass
from tflite2xcore.xcore_schema import XCOREOpCodes

from ..model_builders import build_padded_DW
from ..test_conv2d_passes.conftest import (
    PARAMS as CONV_PARAMS,
    _test_non_matching_params,
    test_matching_params,
)
from .conftest import PARAMS, update_params_with_paddings


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

CONV_PARAMS = deepcopy(CONV_PARAMS)
PARAMS = deepcopy(PARAMS)

# NOTE: this is intentional to keep test case count lower
PARAMS["extended"].update(CONV_PARAMS["default"])
PARAMS["default"].update(CONV_PARAMS["smoke"])
PARAMS["smoke"].update(CONV_PARAMS["smoke"])

PARAMS = update_params_with_paddings(
    PARAMS, is_matching=lambda padding: padding[0] == padding[3] == [0, 0]
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model():
    return build_padded_DW


@pytest.fixture()
def trf_pass():
    return FuseConv2dPaddingPass()


@pytest.fixture()
def weight_shape(kernel_height, kernel_width, input_channels):
    return [kernel_height, kernel_width, input_channels]


@pytest.fixture()
def model(weight_shape, input_size, paddings, strides):
    return build_padded_DW(
        weight_shape=weight_shape,
        input_size=input_size,
        paddings=paddings,
        strides=strides,
    )


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(trf_pass, model):
    # extract original model info
    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 2
    pad_params_pad_ori = subgraph.operators[0].inputs[1].numpy.tolist()
    pad_params_conv_ori = subgraph.operators[-1].custom_options["pad"]
    in_ori, out_ori = subgraph.inputs[0], subgraph.outputs[0]
    in_shape_ori, out_shape_ori = deepcopy(in_ori.shape), deepcopy(out_ori.shape)

    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    # need to clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    # check operator
    assert len(subgraph.operators) == 1
    op = subgraph.operators[0]
    assert op.operator_code.code is XCOREOpCodes.XC_conv2d_depthwise
    assert len(op.inputs) == 3
    assert len(op.outputs) == 1

    # check input/output tensors
    assert len(subgraph.inputs) == len(subgraph.outputs) == 1
    in_new, out_new = subgraph.inputs[0], subgraph.outputs[0]
    assert in_ori is in_new is op.inputs[0]
    assert out_ori is out_new is op.outputs[0]
    assert in_shape_ori == in_new.shape
    assert out_shape_ori == out_new.shape

    # check 'pad' parameters
    pad_params_new = op.custom_options["pad"]
    assert len(pad_params_new) == 3
    assert pad_params_conv_ori[-1] == pad_params_new[-1]
    assert pad_params_new[0] - pad_params_conv_ori[0] == pad_params_pad_ori[1][0]
    assert pad_params_new[1] - pad_params_conv_ori[1] == pad_params_pad_ori[2][0]


def test_non_matching_paddings(
    trf_pass, build_model, weight_shape, input_size, strides, non_matching_paddings
):
    model = build_model(
        weight_shape=weight_shape,
        input_size=input_size,
        paddings=non_matching_paddings,
        strides=strides,
    )
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
