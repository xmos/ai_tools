# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import EliminateDeadTensorsPass

from . import count_tensors, add_dangling_tensor


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()  # type:ignore
def trf_pass() -> EliminateDeadTensorsPass:
    return EliminateDeadTensorsPass()


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate_identity(model: XCOREModel, trf_pass: EliminateDeadTensorsPass) -> None:
    num_tensors = count_tensors(model)
    trf_pass.run(model)
    model.sanity_check()
    assert num_tensors == count_tensors(model)


def test_mutate(model: XCOREModel, trf_pass: EliminateDeadTensorsPass) -> None:
    add_dangling_tensor(model)
    num_tensors = count_tensors(model)
    trf_pass.run(model)
    model.sanity_check()
    assert num_tensors == count_tensors(model) + 1


if __name__ == "__main__":
    pytest.main()
