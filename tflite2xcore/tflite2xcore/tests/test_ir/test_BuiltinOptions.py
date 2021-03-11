# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
from typing import List, Dict, Any

from tflite2xcore.xcore_schema import (
    BuiltinOpCodes,
    BuiltinOptions,
)


#  ----------------------------------------------------------------------------
#                              PARAMETER VALUES
#  ----------------------------------------------------------------------------

_BUILTIN_OP_CODE_NONES = [
    BuiltinOpCodes.CUSTOM,
    BuiltinOpCodes.DELEGATE,
    BuiltinOpCodes.EMBEDDING_LOOKUP,
    BuiltinOpCodes.FLOOR,
    BuiltinOpCodes.CEIL,
    BuiltinOpCodes.HASHTABLE_LOOKUP,
    BuiltinOpCodes.LOGISTIC,
    BuiltinOpCodes.RELU,
    BuiltinOpCodes.RELU_N1_TO_1,
    BuiltinOpCodes.RELU6,
    BuiltinOpCodes.ROUND,
    BuiltinOpCodes.TANH,
    BuiltinOpCodes.PRELU,
    BuiltinOpCodes.SIN,
    BuiltinOpCodes.LOG,
    BuiltinOpCodes.SQRT,
    BuiltinOpCodes.RSQRT,
    BuiltinOpCodes.ELU,
]

_BUILTIN_OP_CODE_REDUCERS = [
    BuiltinOpCodes.MEAN,
    BuiltinOpCodes.REDUCE_MAX,
    BuiltinOpCodes.REDUCE_MIN,
    BuiltinOpCodes.REDUCE_PROD,
    BuiltinOpCodes.REDUCE_ANY,
    BuiltinOpCodes.SUM,
]

_BUILTIN_OP_CODE_POOLS = [
    BuiltinOpCodes.AVERAGE_POOL_2D,
    BuiltinOpCodes.L2_POOL_2D,
    BuiltinOpCodes.MAX_POOL_2D,
]


PARAMS = {
    level: {
        "builtin_op_code": list(BuiltinOpCodes),
        "builtin_option_type": list(BuiltinOptions),
        "builtin_op_code_none": _BUILTIN_OP_CODE_NONES,
        "builtin_op_code_reducer": _BUILTIN_OP_CODE_REDUCERS,
        "builtin_op_code_pool": _BUILTIN_OP_CODE_POOLS,
    }
    for level in ["extended", "default", "smoke"]
}  # type: Dict[str, Dict[str, List[Any]]]


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture
def option_type_map_values() -> List[BuiltinOptions]:
    return [BuiltinOptions.from_BuiltinOpCodes(op_code) for op_code in BuiltinOpCodes]


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_option_type_map(builtin_op_code: BuiltinOpCodes) -> None:
    option_type = BuiltinOptions.from_BuiltinOpCodes(builtin_op_code)
    assert option_type in BuiltinOptions
    if option_type is BuiltinOptions.NONE:
        assert builtin_op_code in _BUILTIN_OP_CODE_NONES
    if option_type is BuiltinOptions.ReducerOptions:
        assert builtin_op_code in _BUILTIN_OP_CODE_REDUCERS
    if option_type is BuiltinOptions.Pool2DOptions:
        assert builtin_op_code in _BUILTIN_OP_CODE_POOLS


def test_option_type_map_values(
    option_type_map_values: List[BuiltinOptions], builtin_option_type: BuiltinOptions
) -> None:
    assert builtin_option_type in option_type_map_values


def test_option_type_map_nones(builtin_op_code_none: BuiltinOpCodes) -> None:
    assert (
        BuiltinOptions.from_BuiltinOpCodes(builtin_op_code_none) is BuiltinOptions.NONE
    )


def test_option_type_map_reducers(builtin_op_code_reducer: BuiltinOpCodes) -> None:
    assert (
        BuiltinOptions.from_BuiltinOpCodes(builtin_op_code_reducer)
        is BuiltinOptions.ReducerOptions
    )


def test_option_type_map_pools(builtin_op_code_pool: BuiltinOpCodes) -> None:
    assert (
        BuiltinOptions.from_BuiltinOpCodes(builtin_op_code_pool)
        is BuiltinOptions.Pool2DOptions
    )


if __name__ == "__main__":
    pytest.main()
