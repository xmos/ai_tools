# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Optional, Tuple

from tflite2xcore.xcore_schema import XCOREOpCodes, BuiltinOpCodes

from . import ChannelPreservingOpTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output as _test_output,
    test_converted_single_op_model,
    test_reference_model_regression,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class GlobalAveragePooling2dTestModelGenerator(ChannelPreservingOpTestModelGenerator):
    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        kwargs = {"input_shape": input_shape} if input_shape else {}
        return tf.keras.layers.GlobalAveragePooling2D(**kwargs)


GENERATOR = GlobalAveragePooling2dTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_avgpool2d_global


@pytest.fixture
def reference_op_code() -> BuiltinOpCodes:
    return BuiltinOpCodes.MEAN


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------

# TODO: fix this
def test_output(compared_outputs, request):
    name = request.node.name
    if tf.version.VERSION[:3] in ("2.4", "2.5"):
        if (
            name.endswith("[CONFIGS[14]]")
            or name.endswith("[CONFIGS[16]]")
            or name.endswith("[CONFIGS[21]]")
        ):
            request.applymarker(pytest.mark.xfail(run=False))
    _test_output(compared_outputs, request)


if __name__ == "__main__":
    pytest.main()
