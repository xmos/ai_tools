# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore

from ..test_conv2d import converted_op_code, Conv2dTestModelGenerator
from . import (
    ExplicitPaddingMixin,
    test_output,
    test_converted_single_op_model,
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PaddedConv2dTestModelGenerator(ExplicitPaddingMixin, Conv2dTestModelGenerator):
    pass


GENERATOR = PaddedConv2dTestModelGenerator


if __name__ == "__main__":
    pytest.main()
