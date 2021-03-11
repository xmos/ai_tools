# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
from typing import Tuple
from copy import deepcopy

from tflite2xcore.transformation_passes.lce_passes import LegalizeXCBconv2DPaddingPass
from tflite2xcore.xcore_schema import Padding, XCOREOpCodes
from tflite2xcore.xcore_model import XCOREModel

from . import build_XC_bconv2d, _test_non_matching_params
from . import (  # pylint: disable=unused-import
    PARAMS,
    test_matching_params,
    test_mutate as _test_mutate,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

# NOTE: this is intentional to reduce test counts
PARAMS["extended"] = PARAMS["default"]
PARAMS["default"] = PARAMS["smoke"]

for params in PARAMS.values():
    params["opcode"] = [XCOREOpCodes.XC_bconv2d_int8, XCOREOpCodes.XC_bconv2d_bin]

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> LegalizeXCBconv2DPaddingPass:
    return LegalizeXCBconv2DPaddingPass()


@pytest.fixture()
def model(
    weight_shape: Tuple[int, int, int, int],
    input_size: Tuple[int, int],
    padding: Padding,
    strides: Tuple[int, int],
    opcode: XCOREOpCodes,
) -> XCOREModel:
    return build_XC_bconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        padding=padding,
        strides=strides,
        opcode=opcode,
    )


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(
    trf_pass: LegalizeXCBconv2DPaddingPass,
    model: XCOREModel,
    padding: Padding,
    opcode: XCOREOpCodes,
) -> None:
    subgraph = model.subgraphs[0]
    old_input = subgraph.inputs[0]
    old_output = subgraph.outputs[0]

    _test_mutate(trf_pass, model, opcode)

    operators = subgraph.operators
    bconv2d_op = operators[-1]

    assert "padding" not in bconv2d_op.custom_options
    assert bconv2d_op.operator_code.code is opcode
    assert old_output is bconv2d_op.outputs[0]

    if padding is Padding.VALID:
        assert len(operators) == 1
        assert old_input is bconv2d_op.inputs[0]
    else:
        assert len(operators) == 2

        pad_op = operators[0]
        assert old_input is pad_op.inputs[0]
        assert len(pad_op.inputs) == 2
        assert len(pad_op.outputs) == 1

        intermediate = bconv2d_op.inputs[0]
        assert intermediate is pad_op.outputs[0]

        # check that padding is sane
        paddings = pad_op.inputs[1].as_array().tolist()
        for j, (size, pads, padded_size) in enumerate(
            zip(old_input.shape, paddings, intermediate.shape)
        ):
            assert (
                size + sum(pads) == padded_size
            ), f"incorrect padded size in dimension {j}"


def test_non_matching_legal(
    trf_pass: LegalizeXCBconv2DPaddingPass,
    weight_shape: Tuple[int, int, int, int],
    input_size: Tuple[int, int],
    strides: Tuple[int, int],
    opcode: XCOREOpCodes,
) -> None:
    model = build_XC_bconv2d(
        weight_shape=weight_shape,
        input_size=input_size,
        strides=strides,
        opcode=opcode,
    )

    _test_non_matching_params(trf_pass, model)


if __name__ == "__main__":
    pytest.main()
