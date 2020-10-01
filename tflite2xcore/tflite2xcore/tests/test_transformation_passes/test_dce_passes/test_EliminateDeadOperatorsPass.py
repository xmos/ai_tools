# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import EliminateDeadOperatorsPass

from .conftest import model, count_operators, add_dangling_ops


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture()
def trf_pass():
    return EliminateDeadOperatorsPass()


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate_identity(model, trf_pass):
    num_ops = count_operators(model)
    trf_pass.run(model)
    model.sanity_check()
    assert num_ops == count_operators(model)


def test_mutate(model, trf_pass):
    add_dangling_ops(model)
    num_ops = count_operators(model)
    trf_pass.run(model)
    model.sanity_check()
    assert num_ops == count_operators(model) + 2


if __name__ == "__main__":
    pytest.main()
