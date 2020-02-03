# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from pytest_cases import pytest_parametrize_plus, fixture_ref
from tflite2xcore.transformation_passes import RemoveUnusedBuffersPass
from tflite2xcore.xcore_model import TensorType

# TODO: stop using fixtures for parameters, since it makes model sharing accross test modules difficult
from .test_ReplaceSingleinDeepoutDepthwiseConv2DPass import (
    output_channels, kernel_height, kernel_width,
    input_height, input_width, input_size, input_channels,
    weight_shape, padding, model
)


@pytest.fixture()
def trf_pass():
    return RemoveUnusedBuffersPass()


def test_run_identity(model, trf_pass):
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    assert num_buffers == len(model.buffers)


def test_run_mutating(model, trf_pass):
    model.create_buffer()
    model.create_metadata("dummy")
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    assert num_buffers == len(model.buffers) + 1


def test_run_non_mutating(model, trf_pass):
    model.subgraphs[0].create_tensor(
        'dangling_tensor', TensorType.INT16, [1, 32, 1, 1],
        buffer=model.create_buffer()
    )
    model.create_metadata("dummy")
    num_buffers = len(model.buffers)
    trf_pass.run(model)
    assert num_buffers == len(model.buffers)


if __name__ == "__main__":
    pytest.main()
