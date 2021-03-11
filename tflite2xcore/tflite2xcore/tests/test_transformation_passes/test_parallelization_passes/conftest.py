# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from tflite2xcore.transformation_passes import ModelTransformationPass
from tflite2xcore.xcore_model import XCOREModel

from ..conftest import PARAMS, test_matching_params, _test_non_matching_params


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS["extended"].update({"num_threads": [1, 2, 3, 4, 5]})

PARAMS["default"].update({"num_threads": [1, 4, 5]})

PARAMS["smoke"].update({"num_threads": [5]})


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(
    trf_pass: ModelTransformationPass, model: XCOREModel, num_threads: int
) -> None:
    op = model.subgraphs[0].operators[0]
    assert "par" not in op.custom_options

    trf_pass.run(model)
    model.sanity_check()

    _test_non_matching_params(trf_pass, model)
    assert "par" in op.custom_options
