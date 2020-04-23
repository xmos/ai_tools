# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import RemoveUnusedBuffersPass
from tflite2xcore.xcore_schema import TensorType

from .conftest import model, add_dangling_tensor


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return RemoveUnusedBuffersPass()


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_non_matching(model, trf_pass):
    add_dangling_tensor(model)
    model.create_metadata("dummy")
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    model.sanity_check()
    assert num_buffers == len(model.buffers)


def test_mutate_identity(model, trf_pass):
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    model.sanity_check()
    assert num_buffers == len(model.buffers)


def test_mutate(model, trf_pass):
    model.create_buffer()
    model.create_metadata("dummy")
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    model.sanity_check()
    assert num_buffers == len(model.buffers) + 1


if __name__ == "__main__":
    pytest.main()
