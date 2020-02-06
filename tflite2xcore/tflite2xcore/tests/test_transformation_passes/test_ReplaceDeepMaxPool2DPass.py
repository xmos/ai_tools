# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ReplaceDeepMaxPool2DPass

from .model_builders import build_maxpool


MATCHING_INPUT_HEIGHT = [2, 4, 8, 12, 16, 24]
MATCHING_INPUT_WIDTH = MATCHING_INPUT_HEIGHT
MATCHING_INPUT_CHANNELS = [32, 64, 96]
MATCHING_PADDING = ['SAME', 'VALID']

NON_MATCHING_INPUT_HEIGHT = [1, 3, 5, 7, 13, 23]
NON_MATCHING_INPUT_WIDTH = NON_MATCHING_INPUT_HEIGHT
NON_MATCHING_INPUT_CHANNELS = [1, 2, 3, 4, 7, 16, 31, 65]
NON_MATCHING_OPTIONS = ('option', 'value'), [
    ('fused_activation_function', 'RELU'),
    ('fused_activation_function', 'RELU6'),
    ('stride_w', 1),
    ('stride_w', 3),
    ('stride_h', 1),
    ('stride_h', 3),
    ('filter_width', 3),
    ('filter_width', 5),
    ('filter_height', 3),
    ('filter_height', 5)
]


@pytest.fixture()
def trf_pass():
    return ReplaceDeepMaxPool2DPass()


@pytest.fixture(params=MATCHING_INPUT_HEIGHT)
def input_height(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_WIDTH)
def input_width(request):
    return request.param


@pytest.fixture(params=MATCHING_INPUT_CHANNELS)
def input_channels(request):
    return request.param


@pytest.fixture()
def input_shape(input_height, input_width, input_channels):
    return [input_height, input_width, input_channels]


@pytest.fixture(params=MATCHING_PADDING)
def padding(request):
    return request.param


@pytest.fixture()
def model(input_shape, padding):
    return build_maxpool(input_shape=input_shape, padding=padding)


def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_height', NON_MATCHING_INPUT_HEIGHT)
def test_non_matching_input_height(trf_pass, input_shape, input_height, padding):
    input_shape[0] = input_height
    model = build_maxpool(input_shape=input_shape, padding=padding)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_width', NON_MATCHING_INPUT_WIDTH)
def test_non_matching_input_width(trf_pass, input_shape, input_width, padding):
    input_shape[1] = input_width
    model = build_maxpool(input_shape=input_shape, padding=padding)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_channels', NON_MATCHING_INPUT_CHANNELS)
def test_non_matching_input_channels(trf_pass, input_shape, input_channels, padding):
    input_shape[2] = input_channels
    model = build_maxpool(input_shape=input_shape, padding=padding)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize(*NON_MATCHING_OPTIONS)
def non_matching_options(model, option, value):
    op = model.subgraphs[0].operators[-1]
    op.builtin_options[option] = value
    assert not trf_pass.match(op)


if __name__ == "__main__":
    pytest.main()
