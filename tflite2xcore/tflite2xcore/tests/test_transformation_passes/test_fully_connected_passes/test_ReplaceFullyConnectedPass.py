# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import ReplaceFullyConnectedPass

from .conftest import PARAMS, test_matching_params, test_non_matching_tensors


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return ReplaceFullyConnectedPass()


if __name__ == "__main__":
    pytest.main()
