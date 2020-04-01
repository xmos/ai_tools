# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

import sys
import os
from pathlib import Path
from enum import Enum
import ctypes

import numpy as np

from tflite2xcore import libtflite2xcore as lib
from tflite2xcore.xcore_model import TensorType


DEFAULT_TENSOR_ARENA_SIZE = 4000000

class XCOREInterpreterStatus(Enum):
    Ok = 0
    Error = 1

class NumpyToTfLiteTensorType(Enum):
    # see tensorflow/tensorflow/lite/c/c_api_internal.h for values
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

        lib.tensors_size.restype = ctypes.c_size_t
        lib.tensors_size.argtypes = [ctypes.c_void_p]

        lib.inputs_size.restype = ctypes.c_size_t
        lib.inputs_size.argtypes = [ctypes.c_void_p]

        lib.input_tensor_index.restype = ctypes.c_size_t
        lib.input_tensor_index.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

        lib.outputs_size.restype = ctypes.c_size_t
        lib.outputs_size.argtypes = [ctypes.c_void_p]

        lib.output_tensor_index.restype = ctypes.c_size_t
        lib.output_tensor_index.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

        lib.set_tensor.restype = ctypes.c_int
        lib.set_tensor.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

        lib.get_tensor.restype = ctypes.c_int
        lib.get_tensor.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

        lib.get_tensor_dims.restype = ctypes.c_int
        lib.get_tensor_dims.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]

        lib.get_tensor_details.restype = ctypes.c_int
        lib.get_tensor_details.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int)]

        lib.invoke.restype = ctypes.c_int
        lib.invoke.argtypes = [ctypes.c_void_p]

        lib.get_error.restype = ctypes.c_size_t
        lib.get_error.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        if model_path:
            with open(model_path, 'rb') as fd:
                model_content = fd.read()

        self._is_allocated = False
        self.obj = lib.new_interpreter()
        status = lib.initialize(self.obj, model_content, len(model_content), arena_size) 
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


    def get_tensor(self, tensor_index):
        tensor_details = self.get_tensor_details()[tensor_index]
        tensor = np.zeros(tensor_details['shape'], dtype=tensor_details['dtype'])
        shape = tensor.ctypes.shape_as(ctypes.c_int)
        type_ = NumpyToTfLiteTensorType[str(tensor.dtype)].value
        data_ptr = tensor.ctypes.data_as(ctypes.c_void_p)
        self._check_status(lib.get_tensor(self.obj, tensor_index, data_ptr, tensor.ndim, shape, type_))
        return tensor

    def get_tensor_details(self):
        tensor_details = []

        tensor_count = lib.tensors_size(self.obj)
        for tensor_index in range(tensor_count):
            # first get the dimensions of the tensor
            tensor_dims = ctypes.c_int()
            self._check_status(lib.get_tensor_dims(self.obj, tensor_index, ctypes.byref(tensor_dims)))
            # allocate buffer for shape
            tensor_shape = (ctypes.c_int * tensor_dims.value)()
            tensor_name_max_len = 1024
            tensor_name = ctypes.create_string_buffer(tensor_name_max_len)
            tensor_type = ctypes.c_int()
            tensor_scale = ctypes.c_float()
            tensor_zero_point = ctypes.c_int()

            self._check_status(lib.get_tensor_details(self.obj, tensor_index, tensor_name, tensor_name_max_len,
                tensor_shape, ctypes.byref(tensor_type), ctypes.byref(tensor_scale),
                ctypes.byref(tensor_zero_point)))
            
            tensor_details.append({
                'index': tensor_index,
                'name': tensor_name.value,
                'shape': np.array(tensor_shape, dtype=np.int32),
                'dtype': TensorType.to_numpy_dtype(tensor_type.value),
                'quantization': (tensor_scale.value, tensor_zero_point.value)
            })

        return tensor_details


    def get_input_details(self):
        input_details = []
        tensor_details = self.get_tensor_details()
        
        inputs_size = lib.inputs_size(self.obj)
        for input_index in range(inputs_size):
            tensor_index = lib.input_tensor_index(self.obj, input_index)
            input_details.append(tensor_details[tensor_index])

        return input_details

    def get_output_details(self):
        output_details = []
        tensor_details = self.get_tensor_details()
        
        outputs_size = lib.outputs_size(self.obj)
        for output_index in range(outputs_size):
            tensor_index = lib.output_tensor_index(self.obj, output_index)
            output_details.append(tensor_details[tensor_index])

        return output_details
