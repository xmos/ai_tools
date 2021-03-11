# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from ..test_conv2d_shallowin import Conv2dShallowinTestModelGenerator
from . import ExplicitlyPaddedConv2dMixin
from ..test_conv2d_shallowin import converted_op_code  # pylint: disable=unused-import
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_idempotence,
    test_reference_model_regression,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PaddedConv2dShallowinTestModelGenerator(
    ExplicitlyPaddedConv2dMixin, Conv2dShallowinTestModelGenerator
):
    pass


GENERATOR = PaddedConv2dShallowinTestModelGenerator


if __name__ == "__main__":
    pytest.main()
