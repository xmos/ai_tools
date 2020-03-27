# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
from pathlib import Path
from enum import Enum
import ctypes

from tflite2xcore import libtflite2xcore as lib


DEFAULT_TENSOR_ARENA_SIZE = 4000000

class XCOREInterpreterStatus(Enum):
    Ok = 0
    Error = 1

class XCOREInterpreter:
    def __init__(self, model_path=None, model_content=None, arena_size=DEFAULT_TENSOR_ARENA_SIZE):
        lib.new_interpreter.restype = ctypes.c_void_p
        lib.new_interpreter.argtypes = None

        lib.delete_interpreter.restype = None
        lib.delete_interpreter.argtypes = [ctypes.c_void_p]

        lib.initialize.restype = ctypes.c_int
        lib.initialize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]

        lib.allocate_tensors.restype = ctypes.c_int
        lib.allocate_tensors.argtypes = [ctypes.c_void_p]

        lib.set_tensor.restype = ctypes.c_int
        lib.set_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

        lib.get_tensor.restype = ctypes.c_int
        lib.get_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

        lib.invoke.restype = ctypes.c_int
        lib.invoke.argtypes = [ctypes.c_void_p]

        if model_path:
            with open(model_path, 'rb') as fd:
                model_content = fd.read()

        self.obj = lib.new_interpreter()
        status = lib.initialize(self.obj, model_content, arena_size) 
        if status == XCOREInterpreterStatus.Error:
            raise RuntimeError('Unable to initialize interpreter')

    def __del__(self):
        lib.delete_interpreter(self.obj)

    def allocate_tensors(self):
        status = lib.allocate_tensors(self.obj)
        if status == XCOREInterpreterStatus.Error:
            raise RuntimeError('Unable to allocate tensors')

    def invoke(self):
        status = lib.invoke(self.obj)
        if status == XCOREInterpreterStatus.Error:
            raise RuntimeError('Unable to invoke')

    def set_tensor(self, tensor_index, value):
        value_ptr = value.ctypes.data_as(ctypes.c_void_p)
        status = lib.set_tensor(self.obj, tensor_index, value_ptr)
        if status == XCOREInterpreterStatus.Error:
            raise RuntimeError('Unable to set tensor')

    def get_tensor(self, tensor_index, value):
        value_ptr = value.ctypes.data_as(ctypes.c_void_p)
        status = lib.get_tensor(self.obj, tensor_index, value_ptr)
        if status == XCOREInterpreterStatus.Error:
            raise RuntimeError('Unable to get tensor')
