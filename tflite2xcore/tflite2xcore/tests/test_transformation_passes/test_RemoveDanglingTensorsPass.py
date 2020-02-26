# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import RemoveDanglingTensorsPass
from tflite2xcore.xcore_model import TensorType

# TODO: stop using fixtures for parameters, since it makes model sharing accross test modules difficult
from .test_ReplaceSingleinDeepoutDepthwiseConv2DPass import (
    output_channels, kernel_height, kernel_width,
    input_height, input_width, input_size, input_channels,
    weight_shape, padding, model
)


@pytest.fixture()
def trf_pass():
    return RemoveDanglingTensorsPass()


def count_tensors(model):
    return sum(len(subgraph.tensors) for subgraph in model.subgraphs)


def test_run_identity(model, trf_pass):
    num_tensors = count_tensors(model)
    trf_pass.run(model)
    model.sanity_check()
    assert num_tensors == count_tensors(model)


def test_run_mutating(model, trf_pass):
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
