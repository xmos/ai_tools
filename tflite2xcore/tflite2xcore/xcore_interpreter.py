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
DEFAULT_XCORE_ARENA_SIZE = 25000


class XCOREInterpreterStatus(Enum):
    Ok = 0
    Error = 1


class NumpyToTfLiteTensorType(Enum):
    # see tensorflow/tensorflow/lite/c/c_api_internal.h for values
    float32 = 1  # kTfLiteFloat32
    int32 = 2  # kTfLiteInt32
    unint8 = 3  # kTfLiteUInt8
    int64 = 4  # kTfLiteInt64
    int16 = 7  # kTfLiteInt16
    complex64 = 7  # kTfLiteComplex64
    int8 = 9  # kTfLiteInt8


class XCOREInterpreter:
    def __init__(
        self,
        model_path=None,
        model_content=None,
        tensor_arena_size=DEFAULT_TENSOR_ARENA_SIZE,
        xcore_arena_size=DEFAULT_XCORE_ARENA_SIZE,
    ):
        self._error_msg = ctypes.create_string_buffer(1024)

        lib.new_interpreter.restype = ctypes.c_void_p
        lib.new_interpreter.argtypes = None

        lib.delete_interpreter.restype = None
        lib.delete_interpreter.argtypes = [ctypes.c_void_p]

        lib.initialize.restype = ctypes.c_int
        lib.initialize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]

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
        lib.set_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        lib.get_tensor.restype = ctypes.c_int
        lib.get_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        lib.get_tensor_details_buffer_sizes.restype = ctypes.c_int
        lib.get_tensor_details_buffer_sizes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]

        lib.get_tensor_details.restype = ctypes.c_int
        lib.get_tensor_details.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
        ]

        lib.get_operator_details_buffer_sizes.restype = ctypes.c_int
        lib.get_operator_details_buffer_sizes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]

        lib.get_operator_details.restype = ctypes.c_int
        lib.get_operator_details.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        lib.invoke.restype = ctypes.c_int
        lib.invoke.argtypes = [ctypes.c_void_p]

        lib.get_error.restype = ctypes.c_size_t
        lib.get_error.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        if model_path:
            with open(model_path, "rb") as fd:
                model_content = fd.read()

        self._is_allocated = False
        self.obj = lib.new_interpreter()
        status = lib.initialize(
            self.obj,
            model_content,
            len(model_content),
            tensor_arena_size,
            xcore_arena_size,
        )
        if status == XCOREInterpreterStatus.Error:
            raise RuntimeError("Unable to initialize interpreter")

    def __del__(self):
        lib.delete_interpreter(self.obj)

    def _verify_allocated(self):
        if not self._is_allocated:
            self.allocate_tensors()

    def _check_status(self, status):
        if status == XCOREInterpreterStatus.Error.value:
            lib.get_error(self.obj, self._error_msg)
            raise RuntimeError(self._error_msg.value.decode("utf-8"))

    def allocate_tensors(self):
        if self._is_allocated:
            return  # NOTE: the TFLu interpreter can not be allocated multiple times
        self._is_allocated = True
        self._check_status(lib.allocate_tensors(self.obj))

    def invoke(self, py_callback=None):
        INVOKE_CALLBACK_FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)

        def c_callback(operator_index):
            # get the dimensions of the operator inputs and outputs
            inputs_size = ctypes.c_size_t()
            outputs_size = ctypes.c_size_t()
            self._check_status(
                lib.get_operator_details_buffer_sizes(
                    self.obj,
                    operator_index,
                    ctypes.byref(inputs_size),
                    ctypes.byref(outputs_size),
                )
            )
            # get the inputs and outputs tensor indices
            operator_name_max_len = 1024
            operator_name = ctypes.create_string_buffer(operator_name_max_len)
            operator_version = ctypes.c_int()
            operator_inputs = (ctypes.c_int * inputs_size.value)()
            operator_outputs = (ctypes.c_int * outputs_size.value)()
            self._check_status(
                lib.get_operator_details(
                    self.obj,
                    operator_index,
                    operator_name,
                    operator_name_max_len,
                    ctypes.byref(operator_version),
                    operator_inputs,
                    operator_outputs,
                )
            )
            # get the details
            tensor_details = self.get_tensor_details()
            operator_details = {
                "index": operator_index,
                "name": operator_name.value.decode("utf-8"),
                "version": operator_version.value,
                "inputs": [
                    tensor_details[input_index] for input_index in operator_inputs
                ],
                "outputs": [
                    tensor_details[output_index] for output_index in operator_outputs
                ],
            }

            py_callback(self, operator_details)

        self._verify_allocated()

        if py_callback:
            cb = INVOKE_CALLBACK_FUNC(c_callback)
        else:
            cb = None

        self._check_status(lib.invoke(self.obj, cb))

    def set_tensor(self, tensor_index, value):
        self._verify_allocated()

        shape = value.ctypes.shape_as(ctypes.c_int)
        type_ = NumpyToTfLiteTensorType[str(value.dtype)].value
        data = value.ctypes.data_as(ctypes.c_void_p)
        self._check_status(
            lib.set_tensor(self.obj, tensor_index, data, value.ndim, shape, type_)
        )

    def get_tensor(self, tensor_index):
        self._verify_allocated()

        tensor_details = self.get_tensor_details()[tensor_index]
        tensor = np.zeros(tensor_details["shape"], dtype=tensor_details["dtype"])
        shape = tensor.ctypes.shape_as(ctypes.c_int)
        type_ = NumpyToTfLiteTensorType[str(tensor.dtype)].value
        data_ptr = tensor.ctypes.data_as(ctypes.c_void_p)
        self._check_status(
            lib.get_tensor(self.obj, tensor_index, data_ptr, tensor.ndim, shape, type_)
        )
        return tensor

    def get_tensor_details(self):
        self._verify_allocated()

        tensor_details = []

        tensor_count = lib.tensors_size(self.obj)
        for tensor_index in range(tensor_count):
            # first get the dimensions of the tensor
            dims_size = ctypes.c_size_t()
            shape_size = ctypes.c_size_t()
            zero_point_size = ctypes.c_size_t()
            self._check_status(
                lib.get_tensor_details_buffer_sizes(
                    self.obj,
                    tensor_index,
                    ctypes.byref(dims_size),
                    ctypes.byref(shape_size),
                    ctypes.byref(zero_point_size),
                )
            )
            # allocate buffer for shape
            tensor_shape = (ctypes.c_int * dims_size.value)()
            tensor_name_max_len = 1024
            tensor_name = ctypes.create_string_buffer(tensor_name_max_len)
            tensor_type = ctypes.c_int()
            tensor_scale = (ctypes.c_float * shape_size.value)()
            tensor_zero_point = (ctypes.c_int32 * zero_point_size.value)()

            self._check_status(
                lib.get_tensor_details(
                    self.obj,
                    tensor_index,
                    tensor_name,
                    tensor_name_max_len,
                    tensor_shape,
                    ctypes.byref(tensor_type),
                    tensor_scale,
                    tensor_zero_point,
                )
            )

            scales = np.array(tensor_scale, dtype=np.float)
            if len(tensor_scale) == 1:
                scales = scales[0]

            zero_points = np.array(tensor_zero_point, dtype=np.int32)
            if len(tensor_scale) == 1:
                zero_points = zero_points[0]

            tensor_details.append(
                {
                    "index": tensor_index,
                    "name": tensor_name.value.decode("utf-8"),
                    "shape": np.array(tensor_shape, dtype=np.int32),
                    "dtype": TensorType.to_numpy_dtype(tensor_type.value),
                    "quantization": (scales, zero_points),
                }
            )

        return tensor_details

    def get_input_details(self):
        self._verify_allocated()

        inputs_size = lib.inputs_size(self.obj)
        input_indices = [
            lib.input_tensor_index(self.obj, input_index)
            for input_index in range(inputs_size)
        ]
        tensor_details = self.get_tensor_details()

        return [tensor_details[input_index] for input_index in input_indices]

    def get_output_details(self):
        self._verify_allocated()

        outputs_size = lib.outputs_size(self.obj)
        output_indices = [
            lib.output_tensor_index(self.obj, output_index)
            for output_index in range(outputs_size)
        ]
        tensor_details = self.get_tensor_details()

        return [tensor_details[output_index] for output_index in output_indices]
