# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest
from copy import deepcopy
from typing import Tuple

from tflite2xcore.transformation_passes.lce_passes import (
    ReplaceBconv2DInt8OutPass,
    ReplaceBconv2DPass,
)
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import XCOREOpCodes, Padding, TensorType

from . import (
    build_lceBconv2d,
    _make_name_type_pairs,
    update_lce_params,
    test_mutate as _test_mutate,
)
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


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(
    trf_pass: ReplaceBconv2DPass, model: XCOREModel, new_opcode: XCOREOpCodes
) -> None:
    subgraph = model.subgraphs[0]
    operators = subgraph.operators
    op = operators[-1]
    strides = op.custom_options["stride_height"], op.custom_options["stride_width"]
    padding = op.custom_options["padding"]

    _test_mutate(trf_pass, model, new_opcode)

    assert len(operators) == 1

    new_op = operators[-1]
    assert "stride" in new_op.custom_options
    assert strides == new_op.custom_options["stride"]
    assert "padding" in new_op.custom_options
    assert padding == new_op.custom_options["padding"]


if __name__ == "__main__":
    pytest.main()
