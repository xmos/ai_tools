# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from copy import deepcopy

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes.reshape_passes import (
    RemoveSubsequentReshapePass,
)
from tflite2xcore.xcore_schema import BuiltinOpCodes

from ..model_builders import build_fc_with_subsequent_reshape
from .conftest import (
    _test_non_matching_params,
    test_matching_params as _test_matching_params,
)
from .test_RemovePrecedingReshapePass import (
    PARAMS,
    test_non_matching_reshape_only,
    # test_non_matching_simple,  # TODO: fix this
    test_mutate,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return RemoveSubsequentReshapePass()


@pytest.fixture()
def model(outputs, reshape):
    return build_fc_with_subsequent_reshape(
        fc_output_shape=reshape["input"], reshaped_output_shape=reshape["output"]
    )


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_matching_params(
    trf_pass: RemoveSubsequentReshapePass, model: XCOREModel
) -> None:
    _test_matching_params(trf_pass, model, op_idx=0)


if __name__ == "__main__":
    pytest.main()
