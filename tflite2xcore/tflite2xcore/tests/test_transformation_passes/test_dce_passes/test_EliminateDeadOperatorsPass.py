# Copyright 2019-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import EliminateDeadOperatorsPass

from . import count_operators, add_dangling_ops


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()  # type:ignore
def trf_pass() -> EliminateDeadOperatorsPass:
    return EliminateDeadOperatorsPass()


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate_identity(
    model: XCOREModel, trf_pass: EliminateDeadOperatorsPass
) -> None:
    num_ops = count_operators(model)
    trf_pass.run(model)
    model.sanity_check()
    assert num_ops == count_operators(model)


def test_mutate(model: XCOREModel, trf_pass: EliminateDeadOperatorsPass) -> None:
    add_dangling_ops(model)
    num_ops = count_operators(model)
    trf_pass.run(model)
    model.sanity_check()
    assert num_ops == count_operators(model) + 2


if __name__ == "__main__":
    pytest.main()
