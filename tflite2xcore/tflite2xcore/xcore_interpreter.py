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

class NumpyToTfLiteTensorType(Enum):
    # see tensorflow/tensorflow/lite/c/c_api_internal.h
    float32 = 1     # kTfLiteFloat32
    int32 = 2       # kTfLiteInt32
    unint8 = 3      # kTfLiteUInt8
    int64 = 4       # kTfLiteInt64
    int16 = 7       # kTfLiteInt16
    complex64 = 7   # kTfLiteComplex64
    int8 = 9        # kTfLiteInt8

class XCOREInterpreter:
    def __init__(self, model_path=None, model_content=None, arena_size=DEFAULT_TENSOR_ARENA_SIZE):
        self._error_msg = ctypes.create_string_buffer(1024)

        lib.new_interpreter.restype = ctypes.c_void_p
        lib.new_interpreter.argtypes = None

        lib.delete_interpreter.restype = None
        lib.delete_interpreter.argtypes = [ctypes.c_void_p]

        lib.initialize.restype = ctypes.c_int
        lib.initialize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]

        lib.allocate_tensors.restype = ctypes.c_int
        lib.allocate_tensors.argtypes = [ctypes.c_void_p]

        lib.set_tensor.restype = ctypes.c_int
        lib.set_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

        lib.get_tensor.restype = ctypes.c_int
        lib.get_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

        lib.invoke.restype = ctypes.c_int
        lib.invoke.argtypes = [ctypes.c_void_p]

        lib.get_error.restype = ctypes.c_int
        lib.get_error.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        if model_path:
            with open(model_path, 'rb') as fd:
                model_content = fd.read()

        self._is_allocated = False
        self.obj = lib.new_interpreter()
        status = lib.initialize(self.obj, model_content, arena_size) 
        if status == XCOREInterpreterStatus.Error:
            raise RuntimeError('Unable to initialize interpreter')

    
    def __del__(self):
        lib.delete_interpreter(self.obj)


    def _check_status(self, status):
        if not self._is_allocated:
            raise RuntimeError('allocate_tensors not called')
        if status == XCOREInterpreterStatus.Error.value:
            lib.get_error(self.obj, self._error_msg)
            raise RuntimeError(self._error_msg.value.decode('utf-8'))


    def allocate_tensors(self):
        self._is_allocated = True
        self._check_status(lib.allocate_tensors(self.obj))


    def invoke(self):
        self._check_status(lib.invoke(self.obj))


    def set_tensor(self, tensor_index, value):
        shape = value.ctypes.shape_as(ctypes.c_int)
        type_ = NumpyToTfLiteTensorType[str(value.dtype)].value
        data = value.ctypes.data_as(ctypes.c_void_p)
        self._check_status(lib.set_tensor(self.obj, tensor_index, data, value.ndim, shape, type_))


    def get_tensor(self, tensor_index, value):
        shape = value.ctypes.shape_as(ctypes.c_int)
        type_ = NumpyToTfLiteTensorType[str(value.dtype)].value
        data_ptr = value.ctypes.data_as(ctypes.c_void_p)
        self._check_status(lib.get_tensor(self.obj, tensor_index, data_ptr, value.ndim, shape, type_))
