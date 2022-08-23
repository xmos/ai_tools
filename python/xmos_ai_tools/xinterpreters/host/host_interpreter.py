# Copyright 2022 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import sys
import ctypes
import numpy as np
from enum import Enum
from pathlib import Path

from xmos_ai_tools.xinterpreters.base.base_interpreter import (
    xcore_tflm_base_interpreter,
)

# DLL path for different platforms
__PARENT_DIR = Path(__file__).parent.absolute()
if sys.platform.startswith("linux"):
    lib_path = str(Path.joinpath(__PARENT_DIR, "libs", "linux", "xtflm_python.so"))
elif sys.platform == "darwin":
    lib_path = str(Path.joinpath(__PARENT_DIR, "libs", "macos", "xtflm_python.dylib"))
else:
    lib_path = str(Path.joinpath(__PARENT_DIR, "libs", "windows", "xtflm_python.dll"))

lib = ctypes.cdll.LoadLibrary(lib_path)

from xmos_ai_tools.xinterpreters.host.exceptions import (
    InterpreterError,
    AllocateTensorsError,
    InvokeError,
    SetTensorError,
    GetTensorError,
    ModelSizeError,
    ArenaSizeError,
    DeviceTimeoutError,
)

MAX_TENSOR_ARENA_SIZE = 10000000


class XTFLMInterpreterStatus(Enum):
    OK = 0
    ERROR = 1


class xcore_tflm_host_interpreter(xcore_tflm_base_interpreter):
    """! The xcore interpreters host class.
    The interpreter to be used on a host, inherits from base interpreter.
    """

    def __init__(self, max_tensor_arena_size=MAX_TENSOR_ARENA_SIZE) -> None:
        """! Host interpreter initializer.
        Sets up functions from the cdll, and calls to cdll function to create a new interpreter.
        """
        self._error_msg = ctypes.create_string_buffer(4096)

        lib.new_interpreter.restype = ctypes.c_void_p
        lib.new_interpreter.argtypes = [
            ctypes.c_size_t,
        ]

        lib.print_memory_plan.restype = None
        lib.print_memory_plan.argtypes = [ctypes.c_void_p]

        lib.delete_interpreter.restype = None
        lib.delete_interpreter.argtypes = [ctypes.c_void_p]

        lib.initialize.restype = ctypes.c_int
        lib.initialize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_char_p,
        ]

        lib.set_input_tensor.restype = ctypes.c_int
        lib.set_input_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        lib.get_output_tensor.restype = ctypes.c_int
        lib.get_output_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        lib.get_input_tensor.restype = ctypes.c_int
        lib.get_input_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        lib.invoke.restype = ctypes.c_int
        lib.invoke.argtypes = [ctypes.c_void_p]

        lib.get_error.restype = ctypes.c_size_t
        lib.get_error.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        lib.arena_used_bytes.restype = ctypes.c_size_t
        lib.arena_used_bytes.argtypes = [
            ctypes.c_void_p,
        ]

        self._max_tensor_arena_size = max_tensor_arena_size
        self._op_states = []

        super().__init__()

    def __enter__(self) -> "xcore_tflm_host_interpreter":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """! Exit calls close function to delete interpreter"""
        self.close()

    def initialise_interpreter(self, model_index=0) -> None:
        """! Interpreter initialiser, initialised interpreter with model and parameters (optional)
        @param model_index  The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        """
        max_model_size = 50000000
        self.obj = lib.new_interpreter(max_model_size)
        currentModel = None
        for model in self.models:
            if model.tile == model_index:
                currentModel = model
        status = lib.initialize(
            self.obj,
            currentModel.model_content,
            len(currentModel.model_content),
            10000000,
            currentModel.params_content,
        )
        if XTFLMInterpreterStatus(status) is XTFLMInterpreterStatus.ERROR:
            raise RuntimeError("Unable to initialize interpreter")

    def set_tensor(self, tensor_index, data, model_index=0) -> None:
        """! Write the input tensor of a model.
        @param data  The blob of data to set the tensor to.
        @param tensor_index  The index of input tensor to target. Defaults to 0.
        @param model_index  The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        """
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        l = len(data)
        l2 = self.get_input_tensor_size(tensor_index)
        if l != l2:
            print("ERROR: mismatching size in set_input_tensor %d vs %d" % (l, l2))

        self._check_status(lib.set_input_tensor(self.obj, tensor_index, data, l))

    def get_tensor(
        self, tensor_index=0, model_index=0, tensor=None
    ) -> "Output tensor data":
        """! Read data from the output tensor of a model.
        @param tensor_index  The index of output tensor to target.
        @param model_index  The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @param tensor  Tensor of correct shape to write into (optional).
        @return  The data that was stored in the output tensor.
        """
        outputs = self.get_output_details(model_index)
        inputs = self.get_input_details(model_index)
        type_ = None
        count = 0
        for output in outputs:
            if tensor_index == output["index"]:
                tensor_details = output
                type_ = "output"
                break
            count = count + 1
        if type_ == None:
            count = 0
            for input_ in inputs:
                if tensor_index == input_["index"]:
                    tensor_details = input_
                    type_ = "input"
                    break
                count = count + 1

        l = self.get_tensor_size(tensor_index)
        if tensor is None:
            tensor = np.zeros(tensor_details["shape"], dtype=tensor_details["dtype"])
        else:
            l2 = len(tensor.tobytes())
            if l2 != l:
                print("ERROR: mismatching size in get_output_tensor %d vs %d" % (l, l2))

        data_ptr = tensor.ctypes.data_as(ctypes.c_void_p)
        if type_ == "input":
            self._check_status(lib.get_input_tensor(self.obj, count, data_ptr, l))
        elif type_ == "output":
            self._check_status(lib.get_output_tensor(self.obj, count, data_ptr, l))
        return tensor

    def get_input_tensor(self, input_index=0, model_index=0) -> "Input tensor data":
        """! Read the data in the input tensor of a model.
        @param input_index  The index of input tensor to target.
        @param tensor Tensor of correct shape to write into (optional).
        @param model_index The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return The data that was stored in the output tensor.
        """
        tensor_details = self.get_input_details(model_index)[input_index]
        tensor = np.zeros(tensor_details["shape"], dtype=tensor_details["dtype"])
        data_ptr = tensor.ctypes.data_as(ctypes.c_void_p)

        l = len(tensor.tobytes())
        self._check_status(lib.get_input_tensor(self.obj, input_index, data_ptr, l))
        return tensor

    def invoke(self) -> None:
        """! Invoke the model and starting inference of the current
        state of the tensors.
        """
        INVOKE_CALLBACK_FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)

        self._check_status(lib.invoke(self.obj))

    def close(self) -> None:
        """! Delete the interpreter.
        @params model_index Defines which interpreter to target in systems with multiple.
        """
        if self.obj:
            lib.delete_interpreter(self.obj)
            self.obj = None
            print(self.obj)

    def tensor_arena_size(self) -> "Size of tensor arena":
        """! Read the size of the tensor arena required.
        @return size of the tensor arena as an integer.
        """
        return lib.arena_used_bytes(self.obj)

    def _check_status(self, status) -> None:
        """! Read a status code and raise an exception.
        @param status Status code.
        """
        if XTFLMInterpreterStatus(status) is XTFLMInterpreterStatus.ERROR:
            lib.get_error(self.obj, self._error_msg)
            raise RuntimeError(self._error_msg.value.decode("utf-8"))

    def print_memory_plan(self) -> None:
        """! Print a plan of memory allocation"""
        lib.print_memory_plan(self.obj)
