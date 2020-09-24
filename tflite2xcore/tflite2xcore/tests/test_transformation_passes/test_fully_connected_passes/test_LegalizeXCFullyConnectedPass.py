# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import numpy as np

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.transformation_passes import (
    ReplaceFullyConnectedPass,
    LegalizeXCFullyConnectedPass,
)

from .conftest import PARAMS


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(model: XCOREModel) -> None:
    # extract original parameters
    subgraph = model.subgraphs[0]
    op = subgraph.operators[0]

    weight_shape_old = op.inputs[1].shape
    assert len(weight_shape_old) == 2
    dim_out, dim_in = weight_shape_old

    bias_shape_old = op.inputs[2].shape
    assert len(bias_shape_old) == 1
    assert bias_shape_old[0] == dim_out

    # run replacement pass
    ReplaceFullyConnectedPass().run(model)
    model.sanity_check()
    assert len(subgraph.operators) == 1

    # run legalization pass
    LegalizeXCFullyConnectedPass().run(model)
    model.sanity_check()
    assert len(subgraph.operators) == 1
    new_op = subgraph.operators[0]
    assert len(new_op.inputs) == 3

    # check weight tensors
    weight_shape_new = new_op.inputs[1].shape
    assert len(weight_shape_new) == 2
    assert weight_shape_new[0] == dim_out
    assert weight_shape_new[1] == int(np.ceil(dim_in / 4)) * 4

    # check bias/scale/offset tensor
    bso_shape = new_op.inputs[2].shape
    assert len(bso_shape) == 3
    assert bso_shape[0] == int(np.ceil(dim_out / 16))
    assert bso_shape[1] == 7
    assert bso_shape[2] == 16


if __name__ == "__main__":
    pytest.main()
