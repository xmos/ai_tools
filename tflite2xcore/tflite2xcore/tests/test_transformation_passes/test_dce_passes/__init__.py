# Copyright 2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore.xcore_schema import TensorType

from tflite2xcore.tests.test_transformation_passes.model_builders import (
    build_tanh,
    _glue_ops,
)

from ..conftest import (  # pylint: disable=unused-import
    _test_non_matching_params,
    test_matching_params,
)

#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------


def count_tensors(model: XCOREModel) -> int:
    return sum(len(subgraph.tensors) for subgraph in model.subgraphs)


def count_operators(model: XCOREModel) -> int:
    return sum(len(subgraph.operators) for subgraph in model.subgraphs)


def add_dangling_tensor(model: XCOREModel) -> None:
    model.subgraphs[0].create_tensor(
        "dangling_tensor", TensorType.INT16, shape=[1, 32, 1, 1]
    )


def add_dangling_ops(model: XCOREModel) -> None:
    subgraph = model.subgraphs[0]
    tin, tout = subgraph.inputs[0], subgraph.outputs[0]

    # add first op
    build_tanh(subgraph, input_shape=tout.shape, tensor_type=tout.type)
    _glue_ops(subgraph.operators[0], subgraph.operators[1])
    dangling_tensor = subgraph.operators[1].outputs[0]

    # add second op
    build_tanh(
        subgraph, input_shape=dangling_tensor.shape, tensor_type=dangling_tensor.type
    )
    _glue_ops(subgraph.operators[1], subgraph.operators[2])

    # fix inputs and outputs
    subgraph.inputs, subgraph.outputs = [tin], [tout]
