# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import (
    RemoveUnusedBuffersPass,
    RemoveDanglingTensorsPass
)
from tflite2xcore.xcore_model import TensorType

from .conftest import model, count_tensors, add_dangling_tensor


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------

def test_mutate(model):
    add_dangling_tensor(model)
    model.create_metadata("dummy")
    num_tensors = count_tensors(model)
    num_buffers = len(model.buffers)

    pass1 = RemoveDanglingTensorsPass()
    pass1.run(model)
    model.sanity_check()

    pass2 = RemoveUnusedBuffersPass()
    pass2.run(model)
    model.sanity_check()

    assert num_tensors == count_tensors(model) + 1
    assert num_buffers == len(model.buffers) + 1


if __name__ == "__main__":
    pytest.main()
