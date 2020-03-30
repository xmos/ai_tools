# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import pytest
from pathlib import Path

import numpy as np

from tflite2xcore.xcore_interpreter import XCOREInterpreter

BUILTIN_OPERATORS_TEST_MODEL = os.path.join(
    Path(__file__).parent.absolute(),
    'builtin_operators.tflite'
)

BUILTIN_OPERATORS_TEST_INPUT = os.path.join(
    Path(__file__).parent.absolute(),
    'test_0.x'
)

BUILTIN_OPERATORS_TEST_OUTPUT = os.path.join(
    Path(__file__).parent.absolute(),
    'test_0.y'
)

def test_model_content():
    with open(BUILTIN_OPERATORS_TEST_MODEL, 'rb') as fd:
        model_content = fd.read()
    interpreter = XCOREInterpreter(model_content=model_content)
    assert(interpreter)

def test_model_path():
    interpreter = XCOREInterpreter(model_path=BUILTIN_OPERATORS_TEST_MODEL)
    assert(interpreter)

def test_inference():
    with open(BUILTIN_OPERATORS_TEST_MODEL, 'rb') as fd:
        model_content = fd.read()
    interpreter = XCOREInterpreter(model_content=model_content)

    interpreter.allocate_tensors()

    input_ = np.fromfile(BUILTIN_OPERATORS_TEST_INPUT, dtype=np.float32)
    input_.shape=(1,4,1,1)
    expected_output = np.fromfile(BUILTIN_OPERATORS_TEST_OUTPUT, dtype=np.float32)
    expected_output.shape=(1,4)
    computed_output = np.zeros(expected_output.shape, dtype=expected_output.dtype)
    
    interpreter.set_tensor(5, input_)
    interpreter.invoke()
    interpreter.get_tensor(6, computed_output)

    np.testing.assert_equal(computed_output, expected_output)

if __name__ == "__main__":
    pytest.main()
