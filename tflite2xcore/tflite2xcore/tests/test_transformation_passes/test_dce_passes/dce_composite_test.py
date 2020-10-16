# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.transformation_passes import (
    EliminateDeadBuffersPass,
    EliminateDeadTensorsPass,
    EliminateDeadOperatorsPass,
)

from . import (
    count_tensors,
    count_operators,
    add_dangling_ops,
    add_dangling_tensor,
)


#  ----------------------------------------------------------------------------
#                               TEST FUNCTIONS
#  ----------------------------------------------------------------------------


def test_mutate(model):
    add_dangling_ops(model)
    add_dangling_tensor(model)
    model.create_metadata("dummy")
    num_ops = count_operators(model)
    num_tensors = count_tensors(model)
    num_buffers = len(model.buffers)

    # add_dangling_ops leaves two dead buffers, remove those first
    pass2 = EliminateDeadBuffersPass()
    pass2.run(model)
    model.sanity_check()

    assert num_ops == count_operators(model)
    assert num_tensors == count_tensors(model)
    assert num_buffers == len(model.buffers) + 2

    # this will remove two dead ops, and leave their outputs dangling
    EliminateDeadOperatorsPass().run(model)
    model.sanity_check()

    assert num_ops == count_operators(model) + 2
    assert num_tensors == count_tensors(model)
    assert num_buffers == len(model.buffers) + 2

    # this cleans up the original danling tensor, plus the other two left by the previous pass
    EliminateDeadTensorsPass().run(model)
    model.sanity_check()

    assert num_ops == count_operators(model) + 2
    assert num_tensors == count_tensors(model) + 3
    assert num_buffers == len(model.buffers) + 2

    # each tensor leaves one dead buffer
    pass2 = EliminateDeadBuffersPass()
    pass2.run(model)
    model.sanity_check()

    assert num_ops == count_operators(model) + 2
    assert num_tensors == count_tensors(model) + 3
    assert num_buffers == len(model.buffers) + 2 + 3


if __name__ == "__main__":
    pytest.main()
