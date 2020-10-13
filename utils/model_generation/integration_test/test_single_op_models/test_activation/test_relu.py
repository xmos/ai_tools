# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore
import tensorflow as tf  # type: ignore
from typing import Callable

from . import LUTActivationOpTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class ReluTestModelGenerator(LUTActivationOpTestModelGenerator):
    @property
    def act_fun(self) -> Callable[[tf.Tensor], tf.Tensor]:
        return lambda x: tf.nn.relu(x)


GENERATOR = ReluTestModelGenerator


if __name__ == "__main__":
    pytest.main()
