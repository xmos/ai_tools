#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved

import sys, os
import numpy as np
import cv2
from math import sqrt

from xmos_ai_tools.xinterpreters import xcore_tflm_usb_interpreter

from xmos_ai_tools import xformer as xf

source_model = "./source_model.tflite"
xcore_model = "./xcore_model.tflite"
xf.convert(source_model, xcore_model, params=None)


def quantize(arr, scale, zero_point, dtype=np.int8):
    t = np.round(arr / scale + zero_point)
    return dtype(np.round(np.clip(t, np.iinfo(dtype).min, np.iinfo(dtype).max)))


def dequantize(arr, scale, zero_point):
    return np.float32((arr.astype(np.int32) - np.int32(zero_point)) * scale)


def imageToInput(path, input_size):
    input_shape_channels = 3
    input_shape_spacial = int(sqrt(input_size / input_shape_channels))
    INPUT_SHAPE = (input_shape_spacial, input_shape_spacial, input_shape_channels)

    # print("Inferred input shape: " + str(INPUT_SHAPE))

    INPUT_SCALE = 0.007843137718737125
    INPUT_ZERO_POINT = -1
    NORM_SCALE = 127.5
    NORM_SHIFT = 1

    OUTPUT_SCALE = 1 / 255.0
    OUTPUT_ZERO_POINT = -128

    ##print("SETTING INPUT TENSOR VIA " + "usb" + "\n")

    import cv2

    img = cv2.imread(path)
    img = cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]))

    # Channel swapping due to mismatch between open CV and XMOS
    img = img[:, :, ::-1]  # or image = image[:, :, (2, 1, 0)]

    img = (img / NORM_SCALE) - NORM_SHIFT
    img = np.round(quantize(img, INPUT_SCALE, INPUT_ZERO_POINT))

    raw_img = bytes(img)
    return raw_img


ie = xcore_tflm_usb_interpreter()
ie.set_model(model_path=xcore_model, model_index=0, secondary_memory=True, flash=False)
raw_img = imageToInput("./goldfish.png", ie.get_input_tensor_size(0, 0))

ie.set_tensor(0, raw_img, 0)
ie.invoke()

out = ie.get_tensor(0, 0)
newout = [abs(x) for x in out]
print(newout)
with open("./out0", "wb") as fd:
    fd.write(bytearray(newout))
