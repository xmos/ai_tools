# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from tflite2xcore.transformation_passes import FloatingPointWarningPass
from tflite2xcore.xcore_schema import XCOREModel, Subgraph, TensorType

_TENSOR_SHAPE = (1, 1, 1, 1)

#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

PARAMS = {
    "default": {
        "tensor_type": [TensorType.FLOAT32, TensorType.FLOAT16, TensorType.FLOAT64],
        "non_matching_tensor_type": [
            TensorType.INT8,
            TensorType.INT16,
            TensorType.INT32,
            TensorType.UINT8,
        ],
    }
}

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass() -> FloatingPointWarningPass:
    return FloatingPointWarningPass()


@pytest.fixture()
def model(tensor_type: TensorType) -> XCOREModel:
    subgraph = Subgraph(model=XCOREModel())
    subgraph.create_tensor("test_tensor", tensor_type, _TENSOR_SHAPE)
    return subgraph.model


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_matching_params(trf_pass: FloatingPointWarningPass, model: XCOREModel) -> None:
    assert trf_pass.match(model.subgraphs[0].tensors[0])


def test_non_matching_tensor_type(
    trf_pass: FloatingPointWarningPass, non_matching_tensor_type: TensorType
) -> None:
    subgraph = Subgraph(model=XCOREModel())
    test_tensor = subgraph.create_tensor(
        "test_tensor", non_matching_tensor_type, _TENSOR_SHAPE
    )
    assert not trf_pass.match(test_tensor)


if __name__ == "__main__":
    pytest.main()
