# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
import itertools

from tflite2xcore.transformation_passes import ReplaceGlobalAveragePool2DPass

from .model_builders import build_mean


from .test_ReplaceAveragePool2DPass import (
    input_height, input_width, input_channels, input_shape,
    NON_MATCHING_INPUT_CHANNELS,
)

MATCHING_REDUCTION_DIMS = (1, 2)
NON_MATCHING_REDUCTION_DIMS = [
    (1,), (2,), (3,), (1, 3), (2, 3)
]

@pytest.fixture()
def trf_pass():
    return ReplaceGlobalAveragePool2DPass()


@pytest.fixture()
def model(input_shape):
    return build_mean(input_shape=input_shape, reduction_dims=MATCHING_REDUCTION_DIMS)


def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_channels', NON_MATCHING_INPUT_CHANNELS)
def test_non_matching_input_channels(trf_pass, input_shape, input_channels):
    input_shape[2] = input_channels
    model = build_mean(input_shape=input_shape, reduction_dims=MATCHING_REDUCTION_DIMS)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('reduction_dims', NON_MATCHING_REDUCTION_DIMS)
def test_non_matching_reduction_dims(trf_pass, input_shape, reduction_dims):
    model = build_mean(input_shape=input_shape, reduction_dims=reduction_dims)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


if __name__ == "__main__":
    pytest.main()
