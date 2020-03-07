# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.operator_codes import XCOREOpCodes
from tflite2xcore.transformation_passes import AddArgMax16OutputPass, ReplaceArgMax16Pass
from .test_AddArgMax16OutputPass import input_shape, model


def test_mutate(model):
    pass1 = AddArgMax16OutputPass()
    pass1.run(model)
    model.sanity_check()

    pass2 = ReplaceArgMax16Pass()
    pass2.run(model)
    model.sanity_check()

    subgraph = model.subgraphs[0]
    assert subgraph.operators[-1].operator_code.code == XCOREOpCodes.XC_argmax_16

    # check input/output/intermediate tensors
    tin = subgraph.get_tensor('input')
    pre_out = subgraph.get_tensor('output')
    tout = subgraph.get_tensor('output_argmax')

    assert len(subgraph.operators) == 2
    assert len(subgraph.tensors) == 3
    assert tin in subgraph.inputs and tin not in subgraph.outputs
    assert pre_out not in (subgraph.inputs + subgraph.outputs)
    assert tout in subgraph.outputs and tout not in subgraph.inputs


if __name__ == "__main__":
    pytest.main()
