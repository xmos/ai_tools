# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from .test_bconv2d_int8 import (  # pylint: disable=unused-import
    GENERATOR,
    RUNNER,
    bitpacked_outputs,
)

from . import (  # pylint: disable=unused-import
    test_output,
)


if __name__ == "__main__":
    pytest.main()
