# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from ..test_conv2d_shallowin import Conv2dShallowinTestModelGenerator
from . import ExplicitlyPaddedConv2dMixin
from . import (  # pylint: disable=unused-import
    test_output,
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
