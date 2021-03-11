# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from typing import Tuple

from tflite2xcore.transformation_passes import ModelTransformationPass

from ..model_builders import build_pad
from ..conftest import _test_non_matching_params, test_replace_mutate

PaddingType = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


def test_non_matching_paddings(
    trf_pass: ModelTransformationPass,
    input_shape: Tuple[int, int, int, int],
    non_matching_paddings: PaddingType,
) -> None:
    model = build_pad(input_shape=input_shape, paddings=non_matching_paddings)
    _test_non_matching_params(trf_pass, model)
