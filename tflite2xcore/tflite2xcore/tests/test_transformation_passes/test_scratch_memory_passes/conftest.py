# Copyright (c) 2020, XMOS Ltd, All rights reserved

from tflite2xcore.transformation_passes import ModelTransformationPass
from tflite2xcore.xcore_model import XCOREModel

from ..conftest import PARAMS, test_matching_params, _test_non_matching_params


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(trf_pass: ModelTransformationPass, model: XCOREModel) -> None:
    op = model.subgraphs[0].operators[0]
    assert "mem" not in op.custom_options

    trf_pass.run(model)
    model.sanity_check()

    _test_non_matching_params(trf_pass, model)
    assert "mem" in op.custom_options
