# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import ScratchMemoryConv2d1x1Pass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_conv2d_1x1,
)

from .conftest import PARAMS


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return ScratchMemoryConv2d1x1Pass()


@pytest.fixture()
def model(weight_shape, input_size, strides):
    return build_XC_conv2d_1x1(
        weight_shape=weight_shape, input_size=input_size, strides=strides
    )


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_matching(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


def test_mutate(trf_pass, model):
    op = model.subgraphs[0].operators[0]
    assert "mem" not in op.custom_options
    trf_pass.run(model)
    model.sanity_check()
    assert "mem" in op.custom_options


if __name__ == "__main__":
    pytest.main()
