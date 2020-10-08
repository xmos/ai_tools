# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest
from copy import deepcopy
from typing import Tuple

from tflite2xcore.transformation_passes.lce_passes import (
    ReplaceBconv2DPass,
    ReplaceBconv2DInt8OutPass,
)
from tflite2xcore.converter import CleanupManager
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import XCOREOpCodes, Padding, TensorType

from . import build_lceBconv2d, _test_non_matching_params, _make_name_type_pairs
from . import (  # pylint: disable=unused-import
    PARAMS,
    test_matching_params,
    test_non_matching_tensors,
    test_non_matching_input_channels,
    test_non_matching_output_channels,
)

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

_NON_MATCHING_TENSORS = list(
    _make_name_type_pairs("output", [TensorType.FLOAT32, TensorType.INT32])
)

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {
        "output_channels": [16, 32, 128],
        "non_matching_output_channels": [4, 7, 8],
        "non_matching_tensors": _NON_MATCHING_TENSORS,
    }
)
PARAMS["default"]["non_matching_tensors"] = _NON_MATCHING_TENSORS[::2]
PARAMS["smoke"]["non_matching_tensors"] = _NON_MATCHING_TENSORS[::2]
for key in (
    "output_channels",
    "non_matching_output_channels",
):
    PARAMS["default"][key] = PARAMS["extended"][key][:-1]
    PARAMS["smoke"][key] = PARAMS["default"][key][:-1]


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ReplaceBconv2DInt8OutPass:
    return ReplaceBconv2DInt8OutPass()


@pytest.fixture()
def model(
    weight_shape: Tuple[int, int, int, int],
    input_size: Tuple[int, int],
    padding: Padding,
    strides: Tuple[int, int],
) -> XCOREModel:
    return build_lceBconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
    )


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(
    trf_pass: ReplaceBconv2DPass,
    model: XCOREModel,
    *,
    custom_opcode: XCOREOpCodes = XCOREOpCodes.XC_bconv2d_int8_out,
) -> None:
    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 1

    # run mutating pass
    trf_pass.run(model)
    model.sanity_check()

    # need to clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    assert len(subgraph.operators) == 1

    assert subgraph.operators[0].operator_code.code is custom_opcode


if __name__ == "__main__":
    pytest.main()
