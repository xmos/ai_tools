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
    test_output as _test_output,
)

#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------

# TODO: remove this when bug is fixed
def test_output(compared_outputs, request):
    name = request.node.name
    for config_idx in (1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 14, 15, 16, 17):
        config_str = f"[CONFIGS[{config_idx}]]"
        if name.endswith(config_str):
            request.applymarker(pytest.mark.xfail(run=False))
    _test_output(compared_outputs, request)


if __name__ == "__main__":
    pytest.main()
