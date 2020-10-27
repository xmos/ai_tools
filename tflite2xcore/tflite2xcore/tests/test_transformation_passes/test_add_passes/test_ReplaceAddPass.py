# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ReplaceAddPass
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import BuiltinOpCodes

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = {"default": {"operator_code": [BuiltinOpCodes.ADD],}}

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> ReplaceAddPass:
    return ReplaceAddPass()


@pytest.fixture()
def model(operator_code: BuiltinOpCodes) -> XCOREModel:
    model = XCOREModel()
    subgraph = model.create_subgraph()
    subgraph.create_operator(operator_code)
    return model


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_matching_params(trf_pass: ReplaceAddPass, model: XCOREModel) -> None:
    assert trf_pass.match(model.subgraphs[0].operators[0])


if __name__ == "__main__":
    pytest.main()
