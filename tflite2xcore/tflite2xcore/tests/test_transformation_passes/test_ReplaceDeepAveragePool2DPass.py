# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ReplaceDeepAveragePool2DPass

from .model_builders import build_avgpool


from .test_ReplaceDeepMaxPool2DPass import (
    input_height, input_width, input_channels, input_shape,
    NON_MATCHING_INPUT_HEIGHT,
    NON_MATCHING_INPUT_WIDTH,
    NON_MATCHING_INPUT_CHANNELS,
    NON_MATCHING_OPTIONS
)


@pytest.fixture()
def trf_pass():
    return ReplaceDeepAveragePool2DPass()


@pytest.fixture()
def model(input_shape):
    return build_avgpool(input_shape=input_shape)


def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_height', NON_MATCHING_INPUT_HEIGHT)
def test_non_matching_input_height(trf_pass, input_shape, input_height):
    input_shape[0] = input_height
    model = build_avgpool(input_shape=input_shape)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_width', NON_MATCHING_INPUT_WIDTH)
def test_non_matching_input_width(trf_pass, input_shape, input_width):
    input_shape[1] = input_width
    model = build_avgpool(input_shape=input_shape)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_channels', NON_MATCHING_INPUT_CHANNELS)
def test_non_matching_input_channels(trf_pass, input_shape, input_channels):
    input_shape[2] = input_channels
    model = build_avgpool(input_shape=input_shape)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize(*NON_MATCHING_OPTIONS)
def non_matching_options(model, option, value):
    op = model.subgraphs[0].operators[-1]
    op.builtin_options[option] = value
    assert not trf_pass.match(op)


if __name__ == "__main__":
    pytest.main()
