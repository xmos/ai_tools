# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest
import numpy as np

from tflite2xcore.utils import VECTOR_SIZE_WORDS, WORD_SIZE_BITS
from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes import (
    LegalizeBconv2dInt8Pass,
    ReplaceBconv2DInt8Pass,
)
from tflite2xcore.transformation_passes.lce_passes import FILLER
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType, XCOREOpCodes

from .test_ReplaceBconv2DInt8Pass import (  # pylint: disable=unused-import
    model,
    new_opcode,
    PARAMS,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()  # type: ignore
def replacement_pass() -> ReplaceBconv2DInt8Pass:
    return ReplaceBconv2DInt8Pass()


@pytest.fixture()  # type: ignore
def legalization_pass() -> LegalizeBconv2dInt8Pass:
    return LegalizeBconv2dInt8Pass()


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(
    replacement_pass: ReplaceBconv2DInt8Pass,
    legalization_pass: LegalizeBconv2dInt8Pass,
    model: XCOREModel,
    new_opcode: XCOREOpCodes,
) -> None:
    subgraph = model.subgraphs[0]

    # run replacement pass
    replacement_pass.mutate(subgraph.operators[0])
    CleanupManager(model).run_passes()

    bconv2d_op = subgraph.operators[0]
    assert bconv2d_op.operator_code.code is new_opcode

    old_weights = bconv2d_op.inputs[1]
    old_multipliers = bconv2d_op.inputs[2]
    old_biases = bconv2d_op.inputs[3]

    # ensure that legalization pass matches
    assert legalization_pass.match(bconv2d_op)

    # run legalization pass
    legalization_pass.mutate(bconv2d_op)
    CleanupManager(model).run_passes()

    # basic checks
    assert len(subgraph.operators) == 1
    assert bconv2d_op is subgraph.operators[0]
    assert len(bconv2d_op.inputs) == 4

    # check custom options
    options = bconv2d_op.custom_options
    assert "illegal_params" not in options
    assert options["K"][:3] == old_weights.shape[:3]
    assert options["K"][3] == old_weights.shape[3] * WORD_SIZE_BITS

    # check multipliers
    new_multipliers = bconv2d_op.inputs[3]
    assert new_multipliers is not old_multipliers
    assert new_multipliers.type is TensorType.INT16
    assert new_multipliers.shape == old_multipliers.shape

    # check biases
    new_biases = bconv2d_op.inputs[3]
    assert new_biases is not old_biases
    assert new_biases.type is TensorType.INT16
    assert new_biases.shape == old_biases.shape

    # check weights
    new_weights = bconv2d_op.inputs[1]
    assert new_weights is not old_weights
    assert new_weights.type is TensorType.INT32
    assert len(new_weights.shape) == 1

    # TODO: find a more precise test
    old_weight_size = np.prod(old_weights.shape)
    assert new_weights.shape[0] - old_weight_size >= 8

    fill = new_weights.as_array().ravel()[old_weight_size:]
    assert np.all(fill == FILLER)


if __name__ == "__main__":
    pytest.main()
