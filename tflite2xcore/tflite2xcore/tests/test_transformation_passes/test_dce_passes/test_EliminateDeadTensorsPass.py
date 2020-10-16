# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import EliminateDeadTensorsPass
from tflite2xcore.xcore_schema import TensorType

from . import count_tensors, add_dangling_tensor


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return EliminateDeadTensorsPass()


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate_identity(model, trf_pass):
    num_tensors = count_tensors(model)
    trf_pass.run(model)
    model.sanity_check()
    assert num_tensors == count_tensors(model)


def test_mutate(model, trf_pass):
    add_dangling_tensor(model)
    num_tensors = count_tensors(model)
    trf_pass.run(model)
    model.sanity_check()
    assert num_tensors == count_tensors(model) + 1


if __name__ == "__main__":
    pytest.main()
