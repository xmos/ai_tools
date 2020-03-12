# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import RemoveUnusedBuffersPass
from tflite2xcore.xcore_model import TensorType

# TODO: use multiple different models instead of just mlp
from .test_ReplaceFullyConnectedIntermediatePass import (
    outputs, hidden_nodes, input_shape, mlp
)


@pytest.fixture()
def trf_pass():
    return RemoveUnusedBuffersPass()


def test_run_identity(mlp, trf_pass):
    model = mlp  # TODO: fix this by refactoring
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    model.sanity_check()
    assert num_buffers == len(model.buffers)


def test_run_mutating(mlp, trf_pass):
    model = mlp  # TODO: fix this by refactoring
    model.create_buffer()
    model.create_metadata("dummy")
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    model.sanity_check()
    assert num_buffers == len(model.buffers) + 1


def test_run_non_mutating(mlp, trf_pass):
    model = mlp  # TODO: fix this by refactoring
    model.subgraphs[0].create_tensor(
        'dangling_tensor', TensorType.INT16, [1, 32, 1, 1],
        buffer=model.create_buffer()
    )
    model.create_metadata("dummy")
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    model.sanity_check()
    assert num_buffers == len(model.buffers)


if __name__ == "__main__":
    pytest.main()
