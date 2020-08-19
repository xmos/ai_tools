# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this

from .. import IntegrationTestModelGenerator, test_output


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_converted_single_op_model(
    xcore_model: XCOREModel, converted_op_code: XCOREOpCodes
) -> None:
    operators = xcore_model.subgraphs[0].operators
    assert len(operators) == 1
    op = operators[0]
    assert op.operator_code.code is converted_op_code
