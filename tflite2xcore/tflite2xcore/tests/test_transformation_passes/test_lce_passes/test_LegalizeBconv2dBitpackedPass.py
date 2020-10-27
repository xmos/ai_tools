# Copyright (c) 2020, XMOS Ltd, All rights reserved
import pytest
import numpy as np

from tflite2xcore.utils import VECTOR_SIZE_WORDS, WORD_SIZE_BITS
from tflite2xcore.converter import CleanupManager
from tflite2xcore.transformation_passes import (
    LegalizeBconv2dBitpackedPass,
    ReplaceBconv2DBitpackedPass,
)
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType, XCOREOpCodes

from .test_ReplaceBconv2DBitpackedPass import (  # pylint: disable=unused-import
    model,
    new_opcode,
    PARAMS,
)


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()  # type: ignore
def replacement_pass() -> ReplaceBconv2DBitpackedPass:
    return ReplaceBconv2DBitpackedPass()


@pytest.fixture()  # type: ignore
def legalization_pass() -> LegalizeBconv2dBitpackedPass:
    return LegalizeBconv2dBitpackedPass()


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_mutate(
    replacement_pass: ReplaceBconv2DBitpackedPass,
    legalization_pass: LegalizeBconv2dBitpackedPass,
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
    old_thresholds = bconv2d_op.inputs[2]

    # ensure that legalization pass matches
    assert legalization_pass.match(bconv2d_op)

    # run legalization pass
    legalization_pass.mutate(bconv2d_op)
    CleanupManager(model).run_passes()

    # basic checks
    assert len(subgraph.operators) == 1
    assert bconv2d_op is subgraph.operators[0]
    assert len(bconv2d_op.inputs) == 3

    # check custom options
    options = bconv2d_op.custom_options
    assert "illegal_params" not in options
    assert options["K"][:3] == old_weights.shape[:3]
    assert options["K"][3] == old_weights.shape[3] * WORD_SIZE_BITS

    # check biases
    new_thresholds = bconv2d_op.inputs[2]
    assert new_thresholds is not old_thresholds
    assert new_thresholds.type is TensorType.INT32
    assert new_thresholds.shape == old_thresholds.shape

    # check weights
    new_weights = bconv2d_op.inputs[1]
    assert new_weights is not old_weights
    assert new_weights.type is TensorType.INT32
    assert len(new_weights.shape) == 1

    kernel_channel_size = np.prod(old_weights.shape[1:])
    filler_size = (
        VECTOR_SIZE_WORDS - kernel_channel_size % VECTOR_SIZE_WORDS
    ) % VECTOR_SIZE_WORDS
    assert new_weights.shape[0] % VECTOR_SIZE_WORDS == filler_size

    if filler_size:
        filler_bits = new_weights.as_array()[-filler_size:]
        assert np.all(
            filler_bits == np.zeros(filler_bits.shape, dtype=filler_bits.dtype)
        )


if __name__ == "__main__":
    pytest.main()
