# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import RemoveDanglingTensorsPass
from tflite2xcore.xcore_model import TensorType

# TODO: use multiple different models instead of just mlp
from .test_ReplaceFullyConnectedIntermediatePass import (
    outputs, hidden_nodes, input_size, mlp
)


@pytest.fixture()
def trf_pass():
    return RemoveDanglingTensorsPass()


def count_tensors(model):
    return sum(len(subgraph.tensors) for subgraph in model.subgraphs)


def test_run_identity(mlp, trf_pass):
    model = mlp  # TODO: fix this by refactoring
    num_tensors = count_tensors(model)
    trf_pass.run(model)
    model.sanity_check()
    assert num_tensors == count_tensors(model)


def test_run_mutating(mlp, trf_pass):
    model = mlp  # TODO: fix this by refactoring
    model.subgraphs[0].create_tensor(
        'dangling_tensor', TensorType.INT16, [1, 32, 1, 1],
        buffer=model.create_buffer()
    )
    num_tensors = count_tensors(model)
    trf_pass.run(model)
    model.sanity_check()
    assert num_tensors == count_tensors(model) + 1


if __name__ == "__main__":
    pytest.main()
