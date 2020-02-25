# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import numpy

from tflite2xcore.xcore_model import TensorType
from tflite2xcore.transformation_passes import ReplaceTanhPass

from ..model_builders import build_tanh
from .conftest import (
    _test_matching_params,
    _test_non_matching_input_type,
    _test_non_matching_output_type,
    _test_mutate
)


@pytest.fixture()
def trf_pass():
    return ReplaceTanhPass()


@pytest.fixture()
def tanh_model(input_shape):
    return build_tanh(input_shape=input_shape, tensor_type=TensorType.INT8)


def test_matching_params(trf_pass, tanh_model):
    _test_matching_params(trf_pass, tanh_model)


def test_non_matching_input_type(trf_pass, tanh_model, non_matching_input_type):
    _test_non_matching_input_type(trf_pass, tanh_model, non_matching_input_type)


def test_non_matching_output_type(trf_pass, tanh_model, non_matching_output_type):
    _test_non_matching_output_type(trf_pass, tanh_model, non_matching_output_type)


def test_mutate(trf_pass, tanh_model):
    _test_mutate(trf_pass, tanh_model)


if __name__ == "__main__":
    pytest.main()
