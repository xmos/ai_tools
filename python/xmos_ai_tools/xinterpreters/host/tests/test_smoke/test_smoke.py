#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved

import sys, os
import numpy as np
import tensorflow as tf

from xmos_ai_tools.xinterpreters import xcore_tflm_host_interpreter

# Load model/params content to test set model arguments
with open("./smoke_model.tflite", "rb") as fd:
    model_content = fd.read()

# Init interpreters
interpreter = tf.lite.Interpreter(
    model_content=model_content,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    experimental_preserve_all_tensors=True,
)
ie = xcore_tflm_host_interpreter()

interpreter.allocate_tensors()
input_tensor_details = interpreter.get_input_details()[0]
input_tensor_type = input_tensor_details["dtype"]
input_tensor_shape = input_tensor_details["shape"]

for test in range(0, 10):
    input_tensor = np.array(
        100 * np.random.random_sample(input_tensor_shape), dtype=input_tensor_type
    )
    interpreter.set_tensor(0, input_tensor)
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])

    ie.set_model(model_path="./smoke_model.tflite")
    ie.set_tensor(0, input_tensor)
    ie.invoke()
    xcore_out_1 = ie.get_tensor(ie.get_output_details()[0]["index"])

    ie.set_model(model_content=model_content)
    ie.set_tensor(0, input_tensor)
    ie.invoke()
    xcore_out_2 = ie.get_tensor(ie.get_output_details()[0]["index"])

    assert tflite_out.all() == xcore_out_1.all()
    assert tflite_out.all() == xcore_out_2.all()


tflite_in_details = interpreter.get_input_details()
tflite_out_details = interpreter.get_output_details()
xcore_in_details = ie.get_input_details()
xcore_out_details = ie.get_output_details()

assert tflite_in_details[0]["name"] == xcore_in_details[0]["name"]
assert tflite_out_details[0]["name"] == xcore_out_details[0]["name"]
assert tflite_in_details[0]["index"] == xcore_in_details[0]["index"]
assert tflite_out_details[0]["index"] == xcore_out_details[0]["index"]
assert tflite_in_details[0]["shape"].all() == xcore_in_details[0]["shape"].all()
assert tflite_out_details[0]["shape"].all() == xcore_out_details[0]["shape"].all()
assert tflite_in_details[0]["dtype"] == xcore_in_details[0]["dtype"]
assert tflite_out_details[0]["dtype"] == xcore_out_details[0]["dtype"]
assert tflite_in_details[0]["quantization"] == xcore_in_details[0]["quantization"]
assert tflite_out_details[0]["quantization"] == xcore_out_details[0]["quantization"]


print(ie.tensor_arena_size())
