# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from copy import deepcopy

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import (
    CanonicalizeSingleinDepthwiseConv2DPass,
    LegalizeSingleinConv2DPass,
)

from .test_ReplaceDepthwiseConv2dPass import (
    weight_shape,
    build_model,
    model,
    test_non_matching_input_channels,
    test_non_matching_depth_multiplier,
)
from .conftest import (
    PARAMS,
    test_matching_params,
    test_non_matching_tensors,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = deepcopy(PARAMS)

for k in PARAMS:
    non_matching_input_channels = [
        c
        for c in PARAMS[k]["input_channels"] + PARAMS[k]["non_matching_input_channels"]
        if c > 1
    ]
    PARAMS[k].update(
        {
            "input_channels": [1],
            "non_matching_input_channels": non_matching_input_channels,
            "depth_multiplier": PARAMS[k]["output_channels"],
            "non_matching_depth_multiplier": PARAMS[k]["non_matching_output_channels"],
        }
    )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> CanonicalizeSingleinDepthwiseConv2DPass:
    return CanonicalizeSingleinDepthwiseConv2DPass()


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(
    trf_pass: CanonicalizeSingleinDepthwiseConv2DPass, model: XCOREModel
) -> None:
    # extract reference data
    subgraph = model.subgraphs[0]
    old_op = subgraph.operators[0]
    old_weight_shape = old_op.inputs[1].shape
    old_bias = old_op.inputs[2]

    # run transformation passes
    trf_pass.run(model)
    model.sanity_check()
    assert len(subgraph.operators) == 1

    LegalizeSingleinConv2DPass().run(model)
    model.sanity_check()
    assert len(subgraph.operators) == 1

    # check operator
    op = subgraph.operators[0]
    assert len(op.inputs) == 3
    assert len(op.outputs) == 1

    # check weight shape
    new_weight_shape = op.inputs[1].shape
    assert new_weight_shape[0] == old_weight_shape[3]
    assert new_weight_shape[1:2] == old_weight_shape[1:2]
    assert new_weight_shape[3] == old_weight_shape[0]

    # check bias
    assert old_bias is op.inputs[2]


if __name__ == "__main__":
    pytest.main()
