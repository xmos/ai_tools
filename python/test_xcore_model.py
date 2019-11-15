#!/usr/bin/env python
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
import ctypes

if sys.platform.startswith("linux"):
    lib = ctypes.cdll.LoadLibrary('../tflite2xcore/build/libtflite2xcore.so')
elif sys.platform == "darwin":
    lib = ctypes.cdll.LoadLibrary('../tflite2xcore/build/libtflite2xcore.dylib')
else:
    lib = ctypes.cdll.LoadLibrary('../tflite2xcore/build/tflite2xcore.dll')

# https://www.auctoris.co.uk/2017/04/29/calling-c-classes-from-python-with-ctypes/

class FlexbufferBuilder(object):
    def __init__(self):
        #lib.new_builder.argtypes = []
        lib.new_builder.restype = ctypes.c_void_p

        lib.builder_int.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.builder_int.restype = ctypes.c_void_p

        lib.builder_finish.argtypes = [ctypes.c_void_p]
        lib.builder_finish.restype = ctypes.c_void_p

        lib.builder_get_buffer.argtypes = [ctypes.c_void_p]
        lib.builder_get_buffer.restype = ctypes.c_char_p

        self.obj = lib.new_builder()
    
    def Int(self, val):
        lib.builder_int(self.obj, 12345)
    
    def Finish(self):
        return lib.builder_finish(self.obj)

    def GetBuffer(self):
        return lib.builder_get_buffer(self.obj)

# model_file = os.path.abspath(b'examples/arm_benchmark/models/model_xcore.tflite')
# model = lib.model_import(model_file)
# print(model)

# subgraphs = lib.model_get_subgraphs(model)
# print(subgraphs)

fbb = FlexbufferBuilder()
fbb.Int(12345)

fbb.Finish()

print('111')
print(fbb.GetBuffer())
