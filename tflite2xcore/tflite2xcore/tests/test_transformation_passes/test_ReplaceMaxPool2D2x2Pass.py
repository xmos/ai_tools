# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ReplaceMaxPool2D2x2Pass

from .model_builders import build_maxpool


from .test_ReplaceMaxPool2DPass import (
    input_channels,
    NON_MATCHING_INPUT_CHANNELS
)

MATCHING_INPUT_HEIGHT = [2, 4, 8, 12, 16, 24]
MATCHING_INPUT_WIDTH = MATCHING_INPUT_HEIGHT
MATCHING_PADDING = ['SAME', 'VALID']
MATCHING_POOL_SIZE = (2, 2)
MATCHING_STRIDES = MATCHING_POOL_SIZE

NON_MATCHING_INPUT_HEIGHT = [1, 3, 5, 7, 13, 23]
NON_MATCHING_INPUT_WIDTH = NON_MATCHING_INPUT_HEIGHT
NON_MATCHING_POOL_SIZE = [
    (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)
]
NON_MATCHING_STRIDES = NON_MATCHING_POOL_SIZE
NON_MATCHING_OPTIONS = ('option', 'value'), [
    ('fused_activation_function', 'RELU'),
    ('fused_activation_function', 'RELU6'),
]


@pytest.fixture()
def trf_pass():
    return ReplaceMaxPool2D2x2Pass()


@pytest.fixture(params=MATCHING_INPUT_HEIGHT)
def input_height(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_WIDTH)
def input_width(request):
    return request.param


@pytest.fixture()
def input_shape(input_height, input_width, input_channels):
    return [input_height, input_width, input_channels]


@pytest.fixture(params=MATCHING_PADDING)
def padding(request):
    return request.param


@pytest.fixture()
def model(input_shape, padding):
    return build_maxpool(input_shape=input_shape, padding=padding,
                         pool_size=MATCHING_POOL_SIZE, strides=MATCHING_STRIDES)


def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_height', NON_MATCHING_INPUT_HEIGHT)
def test_non_matching_input_height(trf_pass, input_shape, input_height, padding):
    input_shape[0] = input_height
    model = build_maxpool(input_shape=input_shape, padding=padding,
                          pool_size=MATCHING_POOL_SIZE, strides=MATCHING_STRIDES)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_width', NON_MATCHING_INPUT_WIDTH)
def test_non_matching_input_width(trf_pass, input_shape, input_width, padding):
    input_shape[1] = input_width
    model = build_maxpool(input_shape=input_shape, padding=padding,
                          pool_size=MATCHING_POOL_SIZE, strides=MATCHING_STRIDES)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_channels', NON_MATCHING_INPUT_CHANNELS)
def test_non_matching_input_channels(trf_pass, input_shape, input_channels, padding):
    input_shape[2] = input_channels
    model = build_maxpool(input_shape=input_shape, padding=padding,
                          pool_size=MATCHING_POOL_SIZE, strides=MATCHING_STRIDES)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('pool_size', NON_MATCHING_POOL_SIZE)
def test_non_matching_pool_size(trf_pass, model, pool_size):
    op = model.subgraphs[0].operators[-1]
    op.builtin_options['filter_height'] = pool_size[0]
    op.builtin_options['filter_width'] = pool_size[1]
    assert not trf_pass.match(op)


@pytest.mark.parametrize('strides', NON_MATCHING_STRIDES)
def test_non_matching_strides(trf_pass, model, strides):
    op = model.subgraphs[0].operators[-1]
    op.builtin_options['stride_h'] = strides[0]
    op.builtin_options['stride_w'] = strides[1]
    assert not trf_pass.match(op)


@pytest.mark.parametrize(*NON_MATCHING_OPTIONS)
def test_non_matching_options(trf_pass, model, option, value):
    op = model.subgraphs[0].operators[-1]
    op.builtin_options[option] = value
    assert not trf_pass.match(op)


if __name__ == "__main__":
    pytest.main()
