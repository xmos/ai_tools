# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def mean_abs_diff_tolerance() -> float:
    return 0.0


@pytest.fixture  # type: ignore
def bitpacked_outputs() -> bool:
    return True
