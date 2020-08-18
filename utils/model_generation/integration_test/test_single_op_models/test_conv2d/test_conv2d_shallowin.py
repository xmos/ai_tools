# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.xcore_schema import XCOREOpCodes  # type: ignore # TODO: fix this
from tflite2xcore._model_generation import Configuration

from . import Conv2dGenericTestModelGenerator, test_output


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Conv2dShallowinTestModelGenerator(Conv2dGenericTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        input_channels = cfg.setdefault("input_channels", 4)
        try:
            assert (
                cfg["K_w"] * input_channels <= 32
            ), "K_w * input_channels > 32 is reserved for general conv2d testing"
            assert (
                cfg["K_h"] != 1 or cfg["K_w"] != 1
            ), "1x1 kernel is reserved for conv2d_1x1 testing"
        except KeyError:
            pass
        super()._set_config(cfg)


GENERATOR = Conv2dShallowinTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_converted_model(xcore_model: XCOREModel) -> None:
    operators = xcore_model.subgraphs[0].operators
    assert len(operators) == 1
    op = operators[0]
    assert op.operator_code.code is XCOREOpCodes.XC_conv2d_shallowin


if __name__ == "__main__":
    pytest.main()
