# Copyright (c) 2020, XMOS Ltd, All rights reserved

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
