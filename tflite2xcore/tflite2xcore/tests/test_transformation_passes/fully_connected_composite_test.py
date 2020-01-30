# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
import numpy

from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes


# TODO: refactor model builders into new file
def build_fc(*, outputs, input_size):
    model = XCOREModel()
    subgraph = model.create_subgraph()

    input_shape = [1] + list(input_size)
    weight_shape = [outputs, numpy.prod(input_shape[1:])]
    tin = subgraph.create_tensor(
        'input', TensorType.INT8, shape=input_shape, isinput=True)
    w = subgraph.create_tensor(
        'weights', TensorType.INT8, shape=weight_shape,
        quantization={'scale': [0.35], 'zero_point': [0]})
    b = subgraph.create_tensor(
        'biases', TensorType.INT32, shape=weight_shape[:1])
    tout = subgraph.create_tensor(
        'output', tin.type, shape=[1, weight_shape[0]], isoutput=True)
    subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.FULLY_CONNECTED),
        inputs=[tin, w, b], outputs=[tout])

    return model


def build_intermediate_fc(*, outputs, input_size):
    model = build_fc(outputs=outputs, input_size=input_size)
    subgraph = model.subgraphs[0]

    subgraph.get_tensor('weights').name = 'weights_1'
    subgraph.get_tensor('biases').name = 'biases_1'

    tmid = subgraph.get_tensor('output')
    tmid.name = 'intermediate'
    subgraph.outputs.remove(tmid)

    return model


def build_mlp(*, outputs, hidden_nodes, input_size):
    model = build_intermediate_fc(outputs=hidden_nodes, input_size=input_size)
    subgraph = model.subgraphs[0]
    tmid = subgraph.get_tensor('intermediate')

    w2_shape = [outputs, hidden_nodes]
    w2 = subgraph.create_tensor('weights_2', TensorType.INT8, shape=w2_shape,
                                quantization={'scale': [0.22], 'zero_point': [0]})
    b2 = subgraph.create_tensor('biases_2', TensorType.INT32, shape=[outputs])
    tout = subgraph.create_tensor('output', tmid.type, shape=[1, outputs], isoutput=True)
    subgraph.create_operator(OperatorCode(BuiltinOpCodes.FULLY_CONNECTED),
                             inputs=[tmid, w2, b2], outputs=[tout])

    return model
