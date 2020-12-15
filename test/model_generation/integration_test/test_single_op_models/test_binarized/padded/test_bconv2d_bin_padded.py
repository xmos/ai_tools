# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from ..test_bconv2d_bin import (  # pylint: disable=unused-import
    GENERATOR,
    RUNNER,
    reference_op_code,
    converted_op_code,
)

from . import (  # pylint: disable=unused-import
    test_reference_model_regression,
    test_converted_model,
    test_mean_abs_diffs,
)


if __name__ == "__main__":
    pytest.main()
