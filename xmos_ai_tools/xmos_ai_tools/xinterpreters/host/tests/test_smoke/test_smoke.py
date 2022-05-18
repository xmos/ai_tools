#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved

import sys, os
import numpy as np
import cv2

from xinterpreters import xcore_tflm_host_interpreter

ie = xcore_tflm_host_interpreter()
ie.set_model(model_path="./smoke_model.tflite", params_path="./smoke_model.flash")
with open("./detection_0.raw", "rb") as fd:
    img = fd.read()

# check that arena usage calcuation is correct
assert ie.tensor_arena_size() == 901376

ie.set_input_tensor(data=img, input_index=0, model_index=0)
ie.invoke()

answer1 = ie.get_output_tensor(output_index=0, model_index=0, tensor=None)
answer2 = ie.get_output_tensor(output_index=1, model_index=0, tensor=None)
with open("./out0", "wb") as fd:
    fd.write(answer1)
with open("./out1", "wb") as fd:
    fd.write(answer2)
