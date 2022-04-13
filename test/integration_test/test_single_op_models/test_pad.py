# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Optional, Tuple

from . import ChannelAgnosticOpTestModelGenerator, PaddingMixin
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PadTestModelGenerator(ChannelAgnosticOpTestModelGenerator, PaddingMixin):
    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        return self._pad_layer(input_shape=input_shape)


GENERATOR = PadTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture
def abs_output_tolerance() -> int:
    return 0

if __name__ == "__main__":
    pytest.main()
