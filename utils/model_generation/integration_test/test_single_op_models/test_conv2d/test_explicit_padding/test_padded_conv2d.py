# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from ..test_conv2d import Conv2dTestModelGenerator
from ..test_conv2d import converted_op_code  # pylint: disable=unused-import
from . import ExplicitPaddingMixin
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_idempotence,
    test_reference_model_regression,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PaddedConv2dTestModelGenerator(ExplicitPaddingMixin, Conv2dTestModelGenerator):
    pass


GENERATOR = PaddedConv2dTestModelGenerator


if __name__ == "__main__":
    pytest.main()
