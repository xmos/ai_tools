# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_schema import ExternalOpCodes, XCOREOpCodes  # type: ignore # TODO: fix this

from .test_bconv2d_int8 import BConv2dInt8TestModelGenerator
from .test_bconv2d_int8 import (  # pylint: disable=unused-import
    bitpacked_outputs,
    RUNNER,
)
from . import (  # pylint: disable=unused-import
    test_reference_model_regression,
    test_converted_single_op_model,
    test_output as _test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class BConv2dInt8DeepInDeepOutTestModelGenerator(BConv2dInt8TestModelGenerator):
    def check_config(self) -> None:
        super().check_config()
        assert (
            self._config["input_channels"] % 256 == 0
        ), "# of input channels must be multiple of 256"
        assert (
            self._config["output_channels"] % 16 == 0
        ), "# of input channels must be multiple of 16"


GENERATOR = BConv2dInt8DeepInDeepOutTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def reference_op_code() -> ExternalOpCodes:
    return ExternalOpCodes.LceBconv2d


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_bconv2d_int8_DIDO


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------

# TODO: remove this when bug is fixed
def test_output(compared_outputs, request):
    name = request.node.name
    for config_idx in (20, 21, 22):
        config_str = f"[CONFIGS[{config_idx}]]"
        if name.endswith(config_str):
            request.applymarker(pytest.mark.xfail(run=False))
    _test_output(compared_outputs, request)


if __name__ == "__main__":
    pytest.main()
