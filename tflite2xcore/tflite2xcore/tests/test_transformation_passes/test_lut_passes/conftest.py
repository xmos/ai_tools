# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
from copy import deepcopy

from tflite2xcore.transformation_passes import LegalizeXCLookupTablePass
from tflite2xcore.xcore_schema import TensorType, XCOREOpCodes

from ..conftest import test_matching_params, _test_non_matching_params


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = {
    "default": {
        "input_channels": [1, 2, 3, 4, 8, 16, 32],
        "input_height": [1, 2, 3, 4, 5, 9],
        "input_width": [1, 2, 3, 4, 5, 9],
        "non_matching_input_type": [
            TensorType.INT16,
            TensorType.INT32,
            TensorType.UINT8,
            TensorType.FLOAT32,
        ],
        "non_matching_output_type": [
            TensorType.INT16,
            TensorType.INT32,
            TensorType.UINT8,
            TensorType.FLOAT32,
        ],
    },
    "smoke": {
        "input_channels": [1, 4, 32],
        "input_height": [1, 9],
        "input_width": [1, 9],
        "non_matching_input_type": [TensorType.INT16, TensorType.FLOAT32],
        "non_matching_output_type": [TensorType.INT16, TensorType.FLOAT32],
    },
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def legalize_table_pass():
    return LegalizeXCLookupTablePass()


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_non_matching_input_type(trf_pass, model, non_matching_input_type):
    op = model.subgraphs[0].operators[0]
    op.inputs[0].type = non_matching_input_type
    _test_non_matching_params(trf_pass, model)


def test_non_matching_output_type(trf_pass, model, non_matching_output_type):
    op = model.subgraphs[0].operators[0]
    op.outputs[0].type = non_matching_output_type
    _test_non_matching_params(trf_pass, model)


def test_mutate(trf_pass, legalize_table_pass, model):
    # extract original parameters
    subgraph = model.subgraphs[0]
    tin_shape = deepcopy(subgraph.get_tensor("input").shape)
    tout_shape = deepcopy(subgraph.get_tensor("output").shape)
    original_opcode = subgraph.operators[0].operator_code.code

    # run replacement pass
    trf_pass.run(model)
    model.sanity_check()

    # check new op
    op = subgraph.operators[-1]
    assert op.operator_code.code == XCOREOpCodes.XC_lookup_8
    assert "original_opcode" in op.custom_options
    assert op.custom_options["original_opcode"] is original_opcode

    # run table legalization pass
    legalize_table_pass.run(model)
    model.sanity_check()
    assert "original_opcode" not in op.custom_options

    # check input/output tensors
    tin = subgraph.get_tensor("input")
    tout = subgraph.get_tensor("output")

    assert len(subgraph.operators) == 1
    assert len(subgraph.tensors) == 3
    assert tin in subgraph.inputs and tin not in subgraph.outputs
    assert tout in subgraph.outputs and tout not in subgraph.inputs
    assert tin.shape == tin_shape
    assert tout.shape == tout_shape

    # check LUT shape
    lut_tensor = op.inputs[1]
    assert len(lut_tensor.buffer.data) == 256
    assert lut_tensor.shape == (256,)
