# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore._model_generation import Configuration

from . import Conv2dGenericTestModelGenerator, test_output


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Conv2d1x1TestModelGenerator(Conv2dGenericTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        assert cfg.setdefault("K_h", 1) == 1, "Kernel height must be 1"
        assert cfg.setdefault("K_w", 1) == 1, "Kernel width must be 1"
        assert cfg.setdefault("strides", (1, 1)) == (1, 1), "strides must be (1, 1)"
        super()._set_config(cfg)


GENERATOR = Conv2d1x1TestModelGenerator


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_converted_model(xcore_model: XCOREModel) -> None:
    operators = xcore_model.subgraphs[0].operators
    assert len(operators) == 1
    op = operators[0]
    assert op.operator_code.code is XCOREOpCodes.XC_conv2d_1x1


if __name__ == "__main__":
    pytest.main()
