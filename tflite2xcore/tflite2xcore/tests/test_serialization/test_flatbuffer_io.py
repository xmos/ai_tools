# Copyright (c) 2019, XMOS Ltd, All rights reserved

import os
import tempfile
import pytest
from pathlib import Path

from tflite2xcore import (
    read_flatbuffer,
    write_flatbuffer
)

from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes, XCOREOpCodes

BUILTIN_OPERATORS_TEST_FILE = os.path.join(
    Path(__file__).parent.absolute(),
    'data/builtin_operators.tflite'
)


def test_read_flatbuffer():
    model = read_flatbuffer(BUILTIN_OPERATORS_TEST_FILE)
    model.pprint()

    assert model.version == 3
    assert len(model.metadata) == 1
    assert len(model.operator_codes) == 6

    assert len(model.buffers) == 19
    assert len(model.buffers[0].data) == 0
    assert len(model.buffers[4].data) == 128

    assert len(model.subgraphs) == 1
    subgraph = model.subgraphs[0]
    assert len(subgraph.operators) == 10
    assert len(subgraph.tensors) == 19
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 1
    assert len(subgraph.intermediates) == len(subgraph.tensors) - len(subgraph.inputs) - len(subgraph.outputs)

    tensor = subgraph.tensors[2]
    assert tensor.name == 'arm_benchmark/conv2d/Conv2D_bias'
    assert tensor.sanitized_name == 'arm_benchmark_conv2d_Conv2D_bias'
    assert tensor.type == TensorType.INT32
    assert tensor.standard_type == 'int32_t'
    assert tensor.shape == [32]
    assert len(tensor.buffer.data) == 128

    operator = subgraph.operators[1]
    assert operator.operator_code.builtin_code == BuiltinOpCodes.CONV_2D
    assert operator.operator_code.version == 3
    assert operator.operator_code.custom_code is None

    assert operator.builtin_options['fused_activation_function'] == 'RELU'
    assert len(operator.inputs) == 3
    assert len(operator.outputs) == 1
    assert operator.outputs[0].name == 'arm_benchmark/re_lu/Relu'


def test_write_flatbuffer():
    model = read_flatbuffer(BUILTIN_OPERATORS_TEST_FILE)

    tmp_file = os.path.join(tempfile.mkdtemp(), 'test_write_flatbuffer.tflite')
    bytes_expected = os.path.getsize(BUILTIN_OPERATORS_TEST_FILE)
    bytes_written = write_flatbuffer(model, tmp_file)

    os.remove(tmp_file)

    assert bytes_written == bytes_expected


def test_custom_options():
    model = XCOREModel()
    subgraph = model.create_subgraph()

    input_tensor = subgraph.create_tensor('input_tensor', TensorType.INT8, [1, 5, 5, 4], isinput=True)
    output_tensor = subgraph.create_tensor('output_tensor', TensorType.INT8, [1, 5, 5, 2], isoutput=True)
    expected_operator = subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_argmax_16),
        inputs=[input_tensor], outputs=[output_tensor]
    )

    expected_operator.custom_options = {
        'int': 1,
        'vector_of_ints': [3, 2, 1],
        'string': 'test string',
        'map': {'one': 1, 'two': 2},
        'vector_of_vectors': [[3, 2, 1], [1, 2, 3], [3, 2, 1]],
        'vector_of_maps ': [
            {'map1': [1, 2, 3]},
            {'map2': [1, 2, 3]},
            {'map3': [1, 2, 3]}
        ]
    }

    tmp_file = os.path.join(tempfile.mkdtemp(), 'test_custom_options.tflite')
    bytes_written = write_flatbuffer(model, tmp_file)

    assert bytes_written > 0

    model = read_flatbuffer(tmp_file)
    model.pprint()

    loaded_operator = model.subgraphs[0].operators[0]
    assert loaded_operator.custom_options == expected_operator.custom_options


if __name__ == "__main__":
    pytest.main()
