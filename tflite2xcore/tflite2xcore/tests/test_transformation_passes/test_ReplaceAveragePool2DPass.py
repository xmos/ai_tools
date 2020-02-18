# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
import itertools

from tflite2xcore.transformation_passes import ReplaceAveragePool2DPass

from .model_builders import build_avgpool


from .test_ReplaceAveragePool2D2x2Pass import (
    input_channels,
    NON_MATCHING_INPUT_CHANNELS,
)

MATCHING_INPUT_HEIGHT = [3, 4, 11]
MATCHING_INPUT_WIDTH = MATCHING_INPUT_HEIGHT
MATCHING_POOL_SIZE = list(itertools.product([1, 2, 3], [1, 2, 3]))
MATCHING_STRIDES = list(itertools.product([1, 2], [1, 2]))
MATCHING_PADDING = 'VALID'

NON_MATCHING_OPTIONS = ('option', 'value'), [
    ('fused_activation_function', 'RELU'),
    ('fused_activation_function', 'RELU6'),
    ('padding', 'SAME')
]


@pytest.fixture()
def trf_pass():
    return ReplaceAveragePool2DPass(safe_mode=False)


@pytest.fixture(params=MATCHING_INPUT_HEIGHT)
def input_height(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_WIDTH)
def input_width(request):
    return request.param


@pytest.fixture(params=MATCHING_POOL_SIZE)
def pool_size(request):
    return request.param


@pytest.fixture(params=MATCHING_STRIDES)
def strides(request):
    return request.param


@pytest.fixture()
def input_shape(input_height, input_width, input_channels):
    return [input_height, input_width, input_channels]


@pytest.fixture()
def model(input_shape, pool_size, strides):
    return build_avgpool(input_shape=input_shape, padding=MATCHING_PADDING,
                         pool_size=pool_size, strides=strides)


def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_channels', NON_MATCHING_INPUT_CHANNELS)
def test_non_matching_input_channels(
        trf_pass, input_shape, input_channels, pool_size, strides):
    input_shape[2] = input_channels
    model = build_avgpool(input_shape=input_shape, padding=MATCHING_PADDING,
                          pool_size=pool_size, strides=strides)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize(*NON_MATCHING_OPTIONS)
def test_non_matching_options(trf_pass, model, option, value):
    op = model.subgraphs[0].operators[-1]
    op.builtin_options[option] = value
    assert not trf_pass.match(op)


if __name__ == "__main__":
    pytest.main()
