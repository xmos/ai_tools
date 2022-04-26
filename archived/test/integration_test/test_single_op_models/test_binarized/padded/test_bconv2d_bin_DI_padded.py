# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from ..test_bconv2d_bin_DI import (  # pylint: disable=unused-import
    GENERATOR,
    RUNNER,
)

from . import (  # pylint: disable=unused-import
    test_mean_abs_diffs,
)


if __name__ == "__main__":
    pytest.main()
