# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import ScratchMemoryConv2dPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_conv2d_deep,
    build_XC_conv2d_shallowin,
    build_XC_conv2d_depthwise,
)

from .conftest import PARAMS

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update(
    {
        "model_builder": [
            build_XC_conv2d_deep,
            build_XC_conv2d_shallowin,
            build_XC_conv2d_depthwise,
        ]
    }
)

PARAMS["smoke"].update(
    {
        "model_builder": [
            build_XC_conv2d_deep,
            build_XC_conv2d_shallowin,
            build_XC_conv2d_depthwise,
        ]
    }
)

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return ScratchMemoryConv2dPass()


@pytest.fixture()
def model(model_builder, weight_shape, input_size, strides):
    return model_builder(
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
