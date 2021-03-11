# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from .test_bconv2d_int8_DIDO import (  # pylint: disable=unused-import
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
