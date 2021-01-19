# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_schema import XCOREModel, Buffer, Metadata
from tflite2xcore.transformation_passes import EliminateDeadBuffersPass

from . import add_dangling_tensor


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()  # type: ignore
def trf_pass() -> EliminateDeadBuffersPass:
    return EliminateDeadBuffersPass()


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_non_matching(model: XCOREModel, trf_pass: EliminateDeadBuffersPass) -> None:
    add_dangling_tensor(model)
    Metadata("dummy", model)
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    model.sanity_check()
    assert num_buffers == len(model.buffers)


def test_mutate_identity(model: XCOREModel, trf_pass: EliminateDeadBuffersPass) -> None:
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    model.sanity_check()
    assert num_buffers == len(model.buffers)


def test_mutate(model: XCOREModel, trf_pass: EliminateDeadBuffersPass) -> None:
    Buffer(model)
    Metadata("dummy", model)
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    model.sanity_check()
    assert num_buffers == len(model.buffers) + 1


if __name__ == "__main__":
    pytest.main()
