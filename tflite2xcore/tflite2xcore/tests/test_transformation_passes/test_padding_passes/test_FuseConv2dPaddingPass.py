# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
from itertools import product
from copy import deepcopy
from typing import Tuple, NamedTuple

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes import FuseConv2dPaddingPass
from tflite2xcore.xcore_schema import XCOREOpCodes
from tflite2xcore.xcore_model import XCOREModel

from ..model_builders import build_padded_DW, _calculate_implicit_pads, ModelBuilder
from ..test_conv2d_passes.conftest import (
    PARAMS as CONV_PARAMS,
    _test_non_matching_params,
    test_matching_params,
)
from .conftest import PARAMS, ParamsType


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

CONV_PARAMS = deepcopy(CONV_PARAMS)
PARAMS = deepcopy(PARAMS)

# NOTE: this is intentional to keep test case count lower
PARAMS["extended"].update(CONV_PARAMS["default"])
PARAMS["default"].update(CONV_PARAMS["smoke"])
PARAMS["smoke"].update(CONV_PARAMS["smoke"])

PaddingType = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]


class ParamCombination(NamedTuple):
    input_channels: int
    input_size: Tuple[int, int]
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: PaddingType


def has_channel_batch_pad(padding: PaddingType) -> bool:
    return not padding[0] == padding[3] == (0, 0)


def has_excessive_padding(param_combo: ParamCombination) -> bool:
    implicit_pads = _calculate_implicit_pads(
        param_combo.stride, param_combo.input_size, param_combo.kernel_size
    )
    effective_pads = tuple(
        tuple(sum(t) for t in zip(*pad_pairs))
        for pad_pairs in zip(implicit_pads, param_combo.padding[1:3])
    )

    for p, k in zip(effective_pads, param_combo.kernel_size):
        if p[0] >= k or p[1] >= k:
            return True
    return False


def update_params_with_paddings(PARAMS: ParamsType) -> ParamsType:
    for params in PARAMS.values():
        input_sizes = product(params["input_height"], params["input_width"])
        kernel_sizes = product(params["kernel_height"], params["kernel_width"])
        strides = product(params["stride_h"], params["stride_w"])
        paddings = product(
            product(params["pad_batch_l"], params["pad_batch_r"]),
            product(params["pad_t"], params["pad_b"]),
            product(params["pad_l"], params["pad_r"]),
            product(params["pad_channel_l"], params["pad_channel_r"]),
        )

        non_matching_channel_batch_params = []
        non_matching_spatial_params = []
        matching_spatial_params = []

        for t in product(
            params["input_channels"], input_sizes, kernel_sizes, strides, paddings
        ):
            param_combo = ParamCombination(*t)
            if has_channel_batch_pad(param_combo.padding):
                non_matching_channel_batch_params.append(param_combo)
            elif has_excessive_padding(param_combo):
                non_matching_spatial_params.append(param_combo)
            else:
                matching_spatial_params.append(param_combo)

        params.update(
            {
                "non_matching_channel_batch_params": non_matching_channel_batch_params,
                "non_matching_spatial_params": non_matching_spatial_params,
                "matching_spatial_params": matching_spatial_params,
            }
        )
    return PARAMS


PARAMS = update_params_with_paddings(PARAMS)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model() -> ModelBuilder:
    return build_padded_DW


@pytest.fixture()
def trf_pass() -> FuseConv2dPaddingPass:
    return FuseConv2dPaddingPass()


@pytest.fixture()
def model(matching_spatial_params: ParamCombination, input_channels: int) -> XCOREModel:
    return build_padded_DW(
        weight_shape=[*matching_spatial_params.kernel_size, input_channels],
        input_size=matching_spatial_params.input_size,
        paddings=matching_spatial_params.padding,
        strides=matching_spatial_params.stride,
    )


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(trf_pass: FuseConv2dPaddingPass, model: XCOREModel) -> None:
    # extract original model info
    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 2
    pad_params_pad_ori = subgraph.operators[0].inputs[1].as_array()
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


def test_non_matching_channel_batch_params(
    trf_pass: FuseConv2dPaddingPass,
    build_model: ModelBuilder,
    non_matching_channel_batch_params: ParamCombination,
    input_channels: int,
) -> None:
    model = build_model(
        weight_shape=[*non_matching_channel_batch_params.kernel_size, input_channels],
        input_size=non_matching_channel_batch_params.input_size,
        paddings=non_matching_channel_batch_params.padding,
        strides=non_matching_channel_batch_params.stride,
    )
    _test_non_matching_params(trf_pass, model)


def test_non_matching_spatial_params(
    trf_pass: FuseConv2dPaddingPass,
    build_model: ModelBuilder,
    non_matching_spatial_params: ParamCombination,
    input_channels: int,
) -> None:
    model = build_model(
        weight_shape=[*non_matching_spatial_params.kernel_size, input_channels],
        input_size=non_matching_spatial_params.input_size,
        paddings=non_matching_spatial_params.padding,
        strides=non_matching_spatial_params.stride,
    )
    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
