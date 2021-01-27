# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from .test_bconv2d_int8 import (  # pylint: disable=unused-import
    GENERATOR,
    RUNNER,
    bitpacked_outputs,
    reference_op_code,
    converted_op_code,
)

from . import (  # pylint: disable=unused-import
    test_reference_model_regression,
    test_converted_single_op_model,
    test_output,
)


if __name__ == "__main__":
    pytest.main()
