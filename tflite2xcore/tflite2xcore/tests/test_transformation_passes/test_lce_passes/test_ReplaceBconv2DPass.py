# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest
from copy import deepcopy

from tflite2xcore.transformation_passes.lce_passes import ReplaceBconv2DPass
from tflite2xcore.converter import CleanupManager
from tflite2xcore.xcore_schema import (
    TensorType,
    XCOREOpCodes,
)

from ..model_builders import build_lceBconv2d
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
    test_non_matching_output_channels,
    test_non_matching_input_channels,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

# NOTE: this is intentional to keep test case count lower
# PARAMS["default"] = PARAMS["smoke"]

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return ReplaceBconv2DPass()


@pytest.fixture()
def model(weight_shape, input_size, padding, strides):
    return build_lceBconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
    )


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(trf_pass, model):

    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 1

    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    # need to clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    assert len(subgraph.operators) == 1

    assert subgraph.operators[0].operator_code.code is XCOREOpCodes.XC_bconv2d_bin_out


if __name__ == "__main__":
    pytest.main()
