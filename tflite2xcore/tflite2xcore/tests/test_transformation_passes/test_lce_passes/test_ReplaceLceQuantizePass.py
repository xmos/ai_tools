# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest

from copy import deepcopy

from tflite2xcore.transformation_passes.lce_passes import ReplaceLceQuantizePass
from tflite2xcore.converter import CleanupManager
from tflite2xcore.xcore_schema import (
    TensorType,
    XCOREOpCodes,
)
from ..model_builders import build_LceQuantize
from .conftest import (
    PARAMS,
    _test_non_matching_params,
    test_matching_params,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

# NOTE: this is intentional to keep test case count lower
PARAMS["default"] = PARAMS["smoke"]

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------
@pytest.fixture()
def trf_pass():
    return ReplaceLceQuantizePass()


@pytest.fixture()
def model(input_shape):
    return build_LceQuantize(input_shape=input_shape,)


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

    assert subgraph.operators[0].operator_code.code is XCOREOpCodes.XC_bsign_8


def test_non_matching_input_tensor(trf_pass, input_shape):

    model = build_LceQuantize(
        input_shape=input_shape, input_tensor_type=TensorType.INT32
    )

    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
