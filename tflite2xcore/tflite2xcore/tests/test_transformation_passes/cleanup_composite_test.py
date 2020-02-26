# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import RemoveUnusedBuffersPass, RemoveDanglingTensorsPass
from tflite2xcore.xcore_model import TensorType

# TODO: use multiple different models instead of just mlp
from .test_RemoveDanglingTensorsPass import (
    outputs, hidden_nodes, input_size, mlp,
    count_tensors
)


@pytest.fixture()
def trf_pass():
    return RemoveUnusedBuffersPass()


def test_mutate(mlp, trf_pass):
    model = mlp  # TODO: fix this by refactoring
    model.subgraphs[0].create_tensor(
        'dangling_tensor', TensorType.INT16, [1, 32, 1, 1],
        buffer=model.create_buffer()
    )
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
