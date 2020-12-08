# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest

from tflite2xcore.transformation_passes import (
    LegalizeBconv2dInt8DeepInDeepOutPass,
    ReplaceBconv2DInt8DeepInDeepOutPass,
)

from .test_LegalizeBconv2dInt8Pass import test_mutate  # pylint: disable=unused-import

from .test_ReplaceBconv2DInt8DeepInDeepOutPass import (  # pylint: disable=unused-import
    model,
    new_opcode,
    PARAMS,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()  # type: ignore
def replacement_pass() -> ReplaceBconv2DInt8DeepInDeepOutPass:
    return ReplaceBconv2DInt8DeepInDeepOutPass()


@pytest.fixture()  # type: ignore
def legalization_pass() -> LegalizeBconv2dInt8DeepInDeepOutPass:
    return LegalizeBconv2dInt8DeepInDeepOutPass()


if __name__ == "__main__":
    pytest.main()
