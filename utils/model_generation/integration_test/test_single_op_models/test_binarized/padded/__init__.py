# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.xcore_schema import (  # type: ignore # TODO: fix this
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


def test_converted_model(
    xcore_model: XCOREModel, converted_op_code: XCOREOpCodes
) -> None:
    operators = xcore_model.subgraphs[0].operators
    assert len(operators) == 2
    op = operators[-1]
    assert op.operator_code.code is converted_op_code
    assert op.inputs[0].producers[0].operator_code.code is XCOREOpCodes.XC_pad
