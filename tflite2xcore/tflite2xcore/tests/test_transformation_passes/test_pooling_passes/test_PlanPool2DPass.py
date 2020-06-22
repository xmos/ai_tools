# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes import PlanPooling2DPass

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_XC_maxpool2d,
    build_XC_avgpool2d,
)

from .test_ReplaceAveragePool2DPass import PARAMS

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["default"].update({"num_threads": [1, 3, 4, 5]})

PARAMS["smoke"].update({"num_threads": [1, 5]})

PARAMS["default"].update({"model_builder": [build_XC_maxpool2d, build_XC_avgpool2d]})


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass(num_threads):
    return PlanPooling2DPass(num_threads=num_threads)


@pytest.fixture()
def model(model_builder, input_shape, pool_size, strides):
    return model_builder(input_shape=input_shape, pool_size=pool_size, strides=strides)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_matching(trf_pass, model, num_threads):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


def test_mutate(trf_pass, model, num_threads):
    op = model.subgraphs[0].operators[0]
    assert "plan" not in op.custom_options
    trf_pass.run(model)
    model.sanity_check()
    assert "plan" in op.custom_options


if __name__ == "__main__":
    pytest.main()
