# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from tflite2xcore.xcore_schema import (
    XCOREOpCodes,
    XCOREModel,
)

from .. import (
    test_reference_model_regression,
    test_mean_abs_diffs,
    test_output,
)

#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


@pytest.mark.skip_on_device  # type: ignore
def test_converted_model(
    xcore_model: XCOREModel, converted_op_code: XCOREOpCodes
) -> None:
    subgraph = xcore_model.subgraphs[0]
    operators = subgraph.operators
    op = operators[-1]
    assert op.operator_code.code is converted_op_code

    if len(operators) == 2:
        padded_input = op.inputs[0]
        assert op.inputs[0].producers[0].operator_code.code is XCOREOpCodes.XC_pad
        assert subgraph.inputs[0].shape != padded_input.shape
