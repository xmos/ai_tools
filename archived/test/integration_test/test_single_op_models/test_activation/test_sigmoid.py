# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Callable

from . import LUTActivationOpTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class SigmoidTestModelGenerator(LUTActivationOpTestModelGenerator):
    @property
    def act_fun(self) -> Callable[[tf.Tensor], tf.Tensor]:
        return lambda x: tf.nn.sigmoid(x)


GENERATOR = SigmoidTestModelGenerator


if __name__ == "__main__":
    pytest.main()
