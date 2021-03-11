# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import numpy as np
from typing import Tuple, Optional

from tflite2xcore.utils import QuantizationTuple
from tflite2xcore.xcore_model import XCOREModel

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    generate_dummy_data,
    build_dequantize,
    build_quantize,
    _glue_ops,
)

from ..conftest import (  # pylint: disable=unused-import
    PARAMS,
    test_matching_params,
    _test_non_matching_params,
)


def build_quantize_dequantize_identity(
    *, input_shape: Tuple[int, ...], quantization: Optional[QuantizationTuple] = None
) -> XCOREModel:
    model = build_dequantize(input_shape=input_shape, input_quantization=quantization)
    subgraph = model.subgraphs[0]
    subgraph.operators[0].inputs[0].buffer.data = generate_dummy_data(
        input_shape, np.int8
    )

    build_quantize(subgraph, input_shape=input_shape, output_quantization=quantization)
    _glue_ops(*subgraph.operators)

    subgraph.inputs = []

    return model
