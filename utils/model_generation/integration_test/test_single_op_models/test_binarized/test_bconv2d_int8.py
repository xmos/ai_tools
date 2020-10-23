# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

pytestmark = pytest.mark.skip  # TODO: remove this

from tflite2xcore.xcore_schema import ExternalOpCodes, XCOREOpCodes  # type: ignore # TODO: fix this

from . import (
    BinarizedTestRunner,
    BConv2dGenericTestModelGenerator,
    LarqConverter,
)

from . import (
    test_reference_model_regression,
    # test_converted_single_op_model,  # TODO: enable this
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


GENERATOR = BConv2dGenericTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


class BConv2dInt8TestRunner(BinarizedTestRunner):
    def make_lce_converter(self) -> LarqConverter:
        return LarqConverter(self, self.get_built_model, strip=True)


RUNNER = BConv2dInt8TestRunner

#  ----------------------------------------------------------------------------
#                                   CONFIGS
#  ----------------------------------------------------------------------------

CONFIGS = {  # TODO: generate random configs
    "default": {
        0: {
            "input_channels": 32,
            "output_channels": 64,
            "K_h": 3,
            "K_w": 3,
            "height": 8,
            "width": 8,
        },
    },
}

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def reference_op_code() -> ExternalOpCodes:
    return ExternalOpCodes.LceBconv2d


@pytest.fixture  # type: ignore
def converted_op_code() -> XCOREOpCodes:
    return XCOREOpCodes.XC_bconv2d_int8


if __name__ == "__main__":
    pytest.main()
