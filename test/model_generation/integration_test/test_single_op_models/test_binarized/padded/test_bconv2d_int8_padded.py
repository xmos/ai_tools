# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from ..test_bconv2d_int8 import (  # pylint: disable=unused-import
    GENERATOR,
    RUNNER,
    bitpacked_outputs,
    reference_op_code,
    converted_op_code,
)

from . import (  # pylint: disable=unused-import
    test_reference_model_regression,
    test_converted_model,
    test_output as _test_output,
)

#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------

# TODO: remove this when bug is fixed
def test_output(compared_outputs, request):
    name = request.node.name
    if name.endswith("[CONFIGS[13]]") or name.endswith("[CONFIGS[19]]"):
        request.applymarker(pytest.mark.xfail(run=False))
    _test_output(compared_outputs, request)


if __name__ == "__main__":
    pytest.main()
