# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ReplaceMaxPool2DPass

from .model_builders import build_maxpool

from .test_ReplaceAveragePool2DPass import (
    input_height, input_width, input_channels, input_shape,
    pool_size, strides,
    MATCHING_PADDING,
    NON_MATCHING_INPUT_CHANNELS,
    NON_MATCHING_OPTIONS
)


@pytest.fixture()
def trf_pass():
    return ReplaceMaxPool2DPass()


@pytest.fixture()
def model(input_shape, pool_size, strides):
    return build_maxpool(input_shape=input_shape, padding=MATCHING_PADDING,
                         pool_size=pool_size, strides=strides)


def test_matching_params(trf_pass, model):
    assert trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_channels', NON_MATCHING_INPUT_CHANNELS)
def test_non_matching_input_channels(
        trf_pass, input_shape, input_channels, pool_size, strides):
    input_shape[2] = input_channels
    model = build_maxpool(input_shape=input_shape, padding=MATCHING_PADDING,
                          pool_size=pool_size, strides=strides)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize(*NON_MATCHING_OPTIONS)
def test_non_matching_options(trf_pass, model, option, value):
    op = model.subgraphs[0].operators[-1]
    op.builtin_options[option] = value
    assert not trf_pass.match(op)


if __name__ == "__main__":
    pytest.main()
