# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import pytest

from tflite2xcore.transformation_passes import (
    LegalizeBconv2dBitpackedDeepInPass,
    ReplaceBconv2DBitpackedDeepInPass,
)

from .test_LegalizeBconv2dBitpackedPass import (  # pylint: disable=unused-import
    test_mutate,
)
from .test_ReplaceBconv2DBitpackedDeepInPass import (  # pylint: disable=unused-import
    model,
    new_opcode,
    PARAMS,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()  # type: ignore
def replacement_pass() -> ReplaceBconv2DBitpackedDeepInPass:
    return ReplaceBconv2DBitpackedDeepInPass()


@pytest.fixture()  # type: ignore
def legalization_pass() -> LegalizeBconv2dBitpackedDeepInPass:
    return LegalizeBconv2dBitpackedDeepInPass()


if __name__ == "__main__":
    pytest.main()
