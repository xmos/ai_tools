# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest


from .test_bconv2d_bin import (
    BConv2dBitpackedTestRunner,
    BConv2dBitpackedTestModelGenerator,
)

from . import (  # pylint: disable=unused-import
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


if __name__ == "__main__":
    pytest.main()
