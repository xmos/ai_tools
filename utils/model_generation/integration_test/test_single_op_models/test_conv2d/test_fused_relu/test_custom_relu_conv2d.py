# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore

from ..test_conv2d import Conv2dTestModelGenerator
from . import FusedCustomReluMixin
from ..test_conv2d import converted_op_code  # pylint: disable=unused-import
from . import (  # pylint: disable=unused-import
    # test_output,  # TODO: enable
    # test_converted_single_op_model,  # TODO: enable
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class CustomReluConv2dTestModelGenerator(
    FusedCustomReluMixin, Conv2dTestModelGenerator
):
    pass


GENERATOR = CustomReluConv2dTestModelGenerator


if __name__ == "__main__":
    pytest.main()
