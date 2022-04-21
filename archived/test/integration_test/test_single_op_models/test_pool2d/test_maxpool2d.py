# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Type

from . import Pool2dGenericTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class MaxPool2dTestModelGenerator(Pool2dGenericTestModelGenerator):
    @property
    def _op_class(self) -> Type[tf.keras.layers.MaxPool2D]:
        return tf.keras.layers.MaxPool2D  # type: ignore


GENERATOR = MaxPool2dTestModelGenerator


if __name__ == "__main__":
    pytest.main()
