# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from . import build_lceBconv2d
from ..test_conv2d_passes.conftest import weight_shape  # pylint: disable=unused-import


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def build_model():
    return build_lceBconv2d


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_non_matching_stride_w(trf_pass, model, non_matching_stride_w):
    op = model.subgraphs[0].operators[0]
    op.custom_options["stride_width"] = non_matching_stride_w
    _test_non_matching_params(trf_pass, model)


def test_non_matching_stride_h(trf_pass, model, non_matching_stride_h):
    op = model.subgraphs[0].operators[0]
    op.custom_options["stride_height"] = non_matching_stride_h
    _test_non_matching_params(trf_pass, model)


def test_non_matching_dilation_w_factor(
    trf_pass, model, non_matching_dilation_w_factor
):
    op = model.subgraphs[0].operators[0]
    op.custom_options["dilation_width_factor"] = non_matching_dilation_w_factor
    _test_non_matching_params(trf_pass, model)


def test_non_matching_dilation_h_factor(
    trf_pass, model, non_matching_dilation_h_factor
):
    op = model.subgraphs[0].operators[0]
    op.custom_options["dilation_height_factor"] = non_matching_dilation_h_factor
    _test_non_matching_params(trf_pass, model)
