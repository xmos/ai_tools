#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved

import sys, os
import numpy as np
import cv2
import tensorflow as tf

from xmos_ai_tools.xinterpreters import xcore_tflm_host_interpreter

#Load model/params content to test set model arguments
with open("./smoke_model.tflite", "rb") as fd:
    model_content = fd.read()

#Init interpreters
interpreter = tf.lite.Interpreter(
            model_content=model_content,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF, 
            experimental_preserve_all_tensors=True)
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

    ie.set_model(model_path="./smoke_model_1.tflite")
    ie.set_tensor(0, input_tensor)
    ie.invoke()
    xcore_out_1 = ie.get_tensor(ie.get_output_details()[0]["index"])

    ie.set_model(model_content=model_content)
    ie.set_tensor(0, input_tensor)
    ie.invoke()
    xcore_out_2 = ie.get_tensor(ie.get_output_details()[0]["index"])

    assert(tflite_out.all() == xcore_out_1.all())
    assert(tflite_out.all() == xcore_out_2.all())





# ie = xcore_tflm_host_interpreter()
# #Set Model combination 1
# ie.set_model(model_path="./smoke_model.tflite", params_path="./smoke_model.params")
# ie.set_tensor(tensor_index=0, data=img, model_index=0)
# ie.invoke()
# assert(out_golden_1 == bytes(ie.get_tensor(tensor_index=0, model_index=0, tensor=None)))
# assert(out_golden_2 == bytes(ie.get_tensor(tensor_index=1, model_index=0, tensor=None)))

# #Set Model combination 2
# ie.set_model(model_path="./smoke_model.tflite", params_content=params_content)
# ie.set_tensor(tensor_index=0, data=img, model_index=0)
# ie.invoke()
# assert(out_golden_1 == bytes(ie.get_tensor(tensor_index=0, model_index=0, tensor=None)))
# assert(out_golden_2 == bytes(ie.get_tensor(tensor_index=1, model_index=0, tensor=None)))

# #Set Model combination 3
# ie.set_model(model_content=model_content, params_path="./smoke_model.params")
# ie.set_tensor(tensor_index=0, data=img, model_index=0)
# ie.invoke()
# assert(out_golden_1 == bytes(ie.get_tensor(tensor_index=0, model_index=0, tensor=None)))
# assert(out_golden_2 == bytes(ie.get_tensor(tensor_index=1, model_index=0, tensor=None)))

# #Set Model combination 4
# ie.set_model(model_content=model_content, params_content=params_content)
# ie.set_tensor(tensor_index=0, data=img, model_index=0)
# ie.invoke()
# assert(out_golden_1 == bytes(ie.get_tensor(tensor_index=0, model_index=0, tensor=None)))
# assert(out_golden_2 == bytes(ie.get_tensor(tensor_index=1, model_index=0, tensor=None)))


# in_details = ie.get_input_details()
# out_details = ie.get_output_details()

# # check that arena usage calcuation is correct
# assert ie.tensor_arena_size() == 901376




