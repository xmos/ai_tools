# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest
from copy import deepcopy
from typing import Tuple

from tflite2xcore.transformation_passes.lce_passes import ReplaceBconv2DBitpackedOutPass
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType, XCOREOpCodes, Padding

from . import build_lceBconv2d, _make_name_type_pairs
from . import (  # pylint: disable=unused-import
    PARAMS,
    test_matching_params,
    test_non_matching_tensors,
    test_non_matching_input_channels,
)
from .test_ReplaceBconv2DInt8OutPass import (  # pylint: disable=unused-import
    test_mutate as _test_mutate,
)

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

_NON_MATCHING_TENSORS = list(
    _make_name_type_pairs("output", [TensorType.FLOAT32, TensorType.INT8])
)

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {
        "output_channels": [32, 128, 256],
        "non_matching_output_channels": [16, 20, 33],
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
def trf_pass() -> ReplaceBconv2DBitpackedOutPass:
    return ReplaceBconv2DBitpackedOutPass()


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
        output_tensor_type=TensorType.INT32,
    )


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(trf_pass: ReplaceBconv2DBitpackedOutPass, model: XCOREModel) -> None:
    _test_mutate(trf_pass, model, custom_opcode=XCOREOpCodes.XC_bconv2d_bin_out)


if __name__ == "__main__":
    pytest.main()
