# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest
from copy import deepcopy
from typing import Tuple

from tflite2xcore.transformation_passes import ReplaceBconv2DBitpackedPass
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType, XCOREOpCodes, Padding

from . import (
    build_lceBconv2d,
    _make_name_type_pairs,
    update_lce_params,
    test_bconv2d_mutate as _test_mutate,
)
from . import (  # pylint: disable=unused-import
    PARAMS,
    test_matching_params,
    test_non_matching_tensors,
    test_non_matching_input_channels,
)

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------


PARAMS = deepcopy(PARAMS)

PARAMS["extended"].update(
    {
        "output_channels": [32, 128, 256],
        "non_matching_output_channels": [16, 20, 33],
        "non_matching_tensors": list(
            _make_name_type_pairs("output", [TensorType.FLOAT32, TensorType.INT8])
        ),
    }
)

PARAMS = update_lce_params(PARAMS)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ReplaceBconv2DBitpackedPass:
    return ReplaceBconv2DBitpackedPass()


@pytest.fixture()
def new_opcode() -> XCOREOpCodes:
    return XCOREOpCodes.XC_bconv2d_bin


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


def test_mutate(
    trf_pass: ReplaceBconv2DBitpackedPass, model: XCOREModel, new_opcode: XCOREOpCodes
) -> None:
    subgraph = model.subgraphs[0]
    operators = subgraph.operators

    _test_mutate(trf_pass, model, new_opcode)

    assert len(operators) == 1

    new_op = operators[-1]

    assert len(new_op.inputs) == 3
    new_op.inputs[1].type is TensorType.INT32
    new_op.inputs[2].type is TensorType.INT32
    new_op.outputs[0].type is TensorType.INT8


if __name__ == "__main__":
    pytest.main()
