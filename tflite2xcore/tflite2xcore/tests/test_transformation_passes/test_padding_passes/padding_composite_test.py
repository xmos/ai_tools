# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the 
# XMOS Public License: Version 1

import pytest

from copy import deepcopy

from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes import (
    SplitPaddingPass,
    FuseConsecutivePadsPass,
    FuseConv2dPaddingPass,
)
from tflite2xcore.xcore_schema import BuiltinOpCodes, XCOREOpCodes

from ..model_builders import build_pad, build_padded_DW
from .test_SplitPaddingPass import (
    is_matching as is_split_matching,
    PARAMS as SPLIT_PARAMS,
)
from .test_FuseConv2dPaddingPass import PARAMS as CONV_PARAMS, has_excessive_padding
from .conftest import PARAMS


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

for k in PARAMS:
    PARAMS[k]["paddings"] = deepcopy(SPLIT_PARAMS[k]["paddings"])
    PARAMS[k]["splittable_spatial_params"] = [
        param_combo
        for param_combo in CONV_PARAMS[k]["non_matching_channel_batch_params"]
        if is_split_matching(param_combo.padding)
        and not has_excessive_padding(param_combo)
    ]

#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_split_fuse_pad(input_shape, paddings):
    model = build_pad(input_shape=input_shape, paddings=paddings)
    operators = model.subgraphs[0].operators
    assert len(operators) == 1
    pad_ori = operators[0].inputs[1]
    paddings_ori = pad_ori.as_array()

    split_pass = SplitPaddingPass()
    split_pass.run(model)
    model.sanity_check()
    assert len(operators) == 2
    op1, op2 = operators
    assert op1.operator_code.code is op2.operator_code.code is BuiltinOpCodes.PAD

    split_pass = FuseConsecutivePadsPass()
    split_pass.run(model)
    model.sanity_check()

    # need to clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    assert len(operators) == 1
    pad_new = operators[0].inputs[1]
    assert pad_new is not pad_ori
    paddings_new = pad_new.as_array()
    assert paddings_new[0][0] == paddings_ori[0][0]
    assert paddings_new[0][1] == paddings_ori[0][1]
    assert paddings_new[1][0] == paddings_ori[1][0]
    assert paddings_new[1][1] == paddings_ori[1][1]
    assert paddings_new[2][0] == paddings_ori[2][0]
    assert paddings_new[2][1] == paddings_ori[2][1]
    assert paddings_new[3][0] == paddings_ori[3][0]
    assert paddings_new[3][1] == paddings_ori[3][1]


def test_split_fuse_conv2d(splittable_spatial_params, input_channels):
    model = build_padded_DW(
        weight_shape=[*splittable_spatial_params.kernel_size, input_channels],
        input_size=splittable_spatial_params.input_size,
        paddings=splittable_spatial_params.padding,
        strides=splittable_spatial_params.stride,
    )
    operators = model.subgraphs[0].operators
    assert len(operators) == 2
    paddings_ori = operators[0].inputs[1].as_array()

    split_pass = SplitPaddingPass()
    split_pass.run(model)
    model.sanity_check()
    assert len(operators) == 3
    op1, op2, op3 = operators
    assert op1.operator_code.code is op2.operator_code.code is BuiltinOpCodes.PAD
    assert op3.operator_code.code is XCOREOpCodes.XC_conv2d_depthwise

    split_pass = FuseConv2dPaddingPass()
    split_pass.run(model)
    model.sanity_check()

    # need to clean up dangling ops/tensors
    CleanupManager(model).run_passes()
    model.sanity_check()

    assert len(operators) == 2
    op1, op2 = operators
    assert op1.operator_code.code is BuiltinOpCodes.PAD
    assert op2.operator_code.code is XCOREOpCodes.XC_conv2d_depthwise
    paddings_new = operators[0].inputs[1].as_array()
    assert paddings_new[1][0] == paddings_new[2][0] == 0
    assert paddings_new[1][1] == paddings_new[2][1] == 0
    assert paddings_new[0][0] == paddings_ori[0][0]
    assert paddings_new[0][1] == paddings_ori[0][1]
    assert paddings_new[3][0] == paddings_ori[3][0]
    assert paddings_new[3][1] == paddings_ori[3][1]


if __name__ == "__main__":
    pytest.main()
