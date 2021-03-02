# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1

import pytest

from tflite2xcore.xcore_schema import XCOREOpCodes


from .test_bconv2d_bin import reference_op_code  # pylint: disable=unused-import
from .test_bconv2d_bin import (
    BConv2dBitpackedTestRunner,
    BConv2dBitpackedTestModelGenerator,
)

from . import (  # pylint: disable=unused-import
    test_reference_model_regression,
    test_converted_single_op_model,
    test_mean_abs_diffs,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class BConv2dBitpackedDeepInTestModelGenerator(BConv2dBitpackedTestModelGenerator):
    def check_config(self) -> None:
        super().check_config()
        assert (
            self._config["input_channels"] % 256 == 0
        ), "# of input channels must be multiple of 256"


GENERATOR = BConv2dBitpackedDeepInTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


RUNNER = BConv2dBitpackedTestRunner

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_bconv2d_bin_DI


if __name__ == "__main__":
    pytest.main()
