# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def output_tolerance() -> int:
    return 1
