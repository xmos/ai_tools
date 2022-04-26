# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from ..test_conv2d import Conv2dTestModelGenerator
from . import ExplicitlyPaddedConv2dMixin
from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PaddedConv2dTestModelGenerator(
    ExplicitlyPaddedConv2dMixin, Conv2dTestModelGenerator
):
    pass


GENERATOR = PaddedConv2dTestModelGenerator


if __name__ == "__main__":
    pytest.main()
