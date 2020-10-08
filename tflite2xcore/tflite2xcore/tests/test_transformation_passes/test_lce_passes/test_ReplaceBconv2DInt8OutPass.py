# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest
from copy import deepcopy
from typing import Tuple

from tflite2xcore.transformation_passes.lce_passes import ReplaceBconv2DInt8OutPass
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import XCOREOpCodes, Padding, TensorType

from . import (
    build_lceBconv2d,
    _make_name_type_pairs,
    update_lce_params,
)
from . import (  # pylint: disable=unused-import
    PARAMS,
    test_matching_params,
    test_non_matching_tensors,
    test_non_matching_input_channels,
    test_non_matching_output_channels,
    test_mutate,
)

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {
        "output_channels": [16, 32, 128],
        "non_matching_output_channels": [4, 7, 8],
        "non_matching_tensors": list(
            _make_name_type_pairs("output", [TensorType.FLOAT32, TensorType.INT32])
        ),
    }
)

PARAMS = update_lce_params(PARAMS)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ReplaceBconv2DInt8OutPass:
    return ReplaceBconv2DInt8OutPass()


@pytest.fixture()
def new_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_bconv2d_int8_out


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


if __name__ == "__main__":
    pytest.main()
