# Copyright 2022 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Type, Optional, Any, List, Tuple, Dict

from numpy import ndarray
from tflite import opcode2name
from tflite.Model import Model
from tflite.TensorType import TensorType
import numpy as np

class XTFLMInterpreterStatus(Enum):
    OK = 0
    ERROR = 1

class xcore_tflm_base_interpreter(ABC):
    """! The xcore interpreters base class.
    Defines a common base interface to be used by the host and device interpreters.
    """

    def __init__(self) -> None:
        """! Base interpreter initializer.
        Initialises the list of models attached to the interpreter.
        """
        self.models: List[xcore_tflm_base_interpreter.modelData] = []

    @abstractmethod
    def initialise_interpreter(self, model_index: int=0) -> None:
        """! Abstract initialising interpreter with model associated with model_index.
        @param model_index The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        """
        raise NotImplementedError

    @abstractmethod
    def set_tensor(self, tensor_index: int, value, model_index: int = 0) -> None:
        """! Abstract method for writing the input tensor of a model.
        @param value  The blob of data to set the tensor to.
        @param tensor_index  The index of input tensor to target. Defaults to 0.
        @param model_index  The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_tensor(self, tensor_index: int = 0, model_index: int = 0, tensor: ndarray = None) -> ndarray:
        """! Abstract method for reading data from the output tensor of a model.
        @param tensor_index  The index of output tensor to target.
        @param model_index  The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @param tensor  Tensor of correct shape to write into (optional).
        @return  The data that was stored in the output tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_input_tensor(self, input_index: int = 0, model_index: int = 0) -> Union[ndarray, List[Union[int, Tuple[float]]]]:
        """! Abstract for reading the data in the input tensor of a model.
        @param input_index  The index of input tensor to target.
        @param model_index The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return The data that was stored in the input tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def invoke(self, model_index: int = 0) -> None:
        """! Abstract method for invoking the model and starting inference of the current
        state of the tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self, model_index: int = 0) -> None:
        """! Abstract method deleting the interpreter.
        @params model_index Defines which interpreter to target in systems with multiple.
        """
        raise NotImplementedError

    @abstractmethod
    def tensor_arena_size(self) -> int:
        """! Abstract method to read the size of the tensor arena required.
        @return size of the tensor arena as an integer.
        """
        raise NotImplementedError

    @abstractmethod
    def _check_status(self, status: XTFLMInterpreterStatus) -> None:
        """! Abstract method to read a status code and raise an exception.
        @param status Status code.
        """
        raise NotImplementedError

    @abstractmethod
    def print_memory_plan(self) -> None:
        """! Abstract method to print a plan of memory allocation"""
        raise NotImplementedError

    def get_input_tensor_size(self, input_index: int = 0, model_index: int = 0) -> int:
        """! Read the size of the input tensor from the model.
        @param input_index  The index of input tensor to target.
        @param model_index The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return The size of the input tensor as an integer.
        """

        # Select correct model from model list
        model = self.get_model(model_index)
        modelBuf = Model.GetRootAsModel(model.model_content, 0)

        # Get index of specific input tensor
        tensorIndex = modelBuf.Subgraphs(0).Inputs(input_index)

        tensorType = modelBuf.Subgraphs(0).Tensors(tensorIndex).Type()

        tensorSize: int
        if tensorType == TensorType.INT8:
            tensorSize = 1  # int8 is 1 byte
        elif tensorType == TensorType.INT32:
            tensorSize = 4  # int32 is 4 bytes
        elif tensorType == TensorType.FLOAT32:
            tensorSize = 4  # float32 is 4 bytes
        else:
            print(tensorType)
            self._check_status(XTFLMInterpreterStatus.ERROR)
            tensorSize = 0

        # Calculate tensor size by multiplying shape elements
        for i in range(0, modelBuf.Subgraphs(0).Tensors(tensorIndex).ShapeLength()):
            tensorSize = tensorSize * modelBuf.Subgraphs(0).Tensors(tensorIndex).Shape(
                i
            )
        return tensorSize

    def get_output_tensor_size(
        self, output_index: int = 0, model_index: int = 0
    ) -> int:
        """! Read the size of the output tensor from the model.
        @param output_index  The index of output tensor to target.
        @param model_index The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return The size of the output tensor as an integer.
        """

        # Select correct model from model list
        modelBuf = None
        model = self.get_model(model_index)
        modelBuf = Model.GetRootAsModel(model.model_content, 0)

        # Get index of specific output tensor
        tensorIndex = modelBuf.Subgraphs(0).Outputs(output_index)

        tensorType = modelBuf.Subgraphs(0).Tensors(tensorIndex).Type()

        tensorSize: int
        if tensorType == TensorType.INT8:
            tensorSize = 1  # int8 is 1 byte
        elif tensorType == TensorType.INT32:
            tensorSize = 4  # int32 is 4 bytes
        elif tensorType == TensorType.FLOAT32:
            tensorSize = 4  # float32 is 4 bytes
        else:
            print(tensorType)
            self._check_status(XTFLMInterpreterStatus.ERROR)
            tensorSize = 0

        # Calculate tensor size by multiplying shape elements
        for i in range(0, modelBuf.Subgraphs(0).Tensors(tensorIndex).ShapeLength()):
            tensorSize = tensorSize * modelBuf.Subgraphs(0).Tensors(tensorIndex).Shape(
                i
            )
        return tensorSize

    def get_tensor_size(self, tensor_index: int = 0, model_index: int = 0) -> int:
        """! Read the size of the input tensor from the model.
        @param tensor_index  The index of input tensor to target.
        @param model_index The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return The size of the input tensor as an integer.
        """

        # Select correct model from model list
        modelBuf = None
        model = self.get_model(model_index)
        modelBuf = Model.GetRootAsModel(model.model_content, 0)

        tensorType = modelBuf.Subgraphs(0).Tensors(tensor_index).Type()
        if tensorType == TensorType.INT8:
            tensorSize = 1  # int8 is 1 byte
        elif tensorType == TensorType.INT32:
            tensorSize = 4  # int32 is 4 bytes
        elif tensorType == TensorType.FLOAT32:
            tensorSize = 4  # float32 is 4 bytes
        else:
            print(tensorType)
            self._check_status(XTFLMInterpreterStatus.ERROR)

        # Calculate tensor size by multiplying shape elements
        for i in range(0, modelBuf.Subgraphs(0).Tensors(tensor_index).ShapeLength()):
            tensorSize = tensorSize * modelBuf.Subgraphs(0).Tensors(tensor_index).Shape(
                i
            )
        return tensorSize

    def get_input_details(self, model_index: int = 0) -> List[Dict[str, Any]]:
        """! Reads the input tensor details from the model.
        @param model_index The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return Tensor details, including the index, name, shape, data type, and quantization
        parameters.
        """

        # Select correct model from model list
        modelBuf = None
        model = self.get_model(model_index)
        modelBuf = Model.GetRootAsModel(model.model_content, 0)

        inputsList = []
        for input_ in range(0, modelBuf.Subgraphs(0).InputsLength()):

            tensorIndex = modelBuf.Subgraphs(0).Inputs(input_)

            # Generate dictioary of tensor details
            dtype: Union[Type[Any]]
            if modelBuf.Subgraphs(0).Tensors(tensorIndex).Type() == TensorType.INT8:
                dtype = np.int8
            elif modelBuf.Subgraphs(0).Tensors(tensorIndex).Type() == TensorType.INT32:
                dtype = np.int32
            elif modelBuf.Subgraphs(0).Tensors(tensorIndex).Type() == TensorType.FLOAT32:
                dtype = np.float32
            else:
                raise Exception

            details = {
                "name": str(modelBuf.Subgraphs(0).Tensors(tensorIndex).Name())[
                    1:
                ].strip("'"),
                "index": tensorIndex,
                "shape": modelBuf.Subgraphs(0).Tensors(tensorIndex).ShapeAsNumpy(),
                "shape_signature": modelBuf.Subgraphs(0)
                .Tensors(tensorIndex)
                .ShapeSignatureAsNumpy(),
                "dtype": dtype,
                "quantization": (
                    modelBuf.Subgraphs(0).Tensors(tensorIndex).Quantization().Scale(0),
                    modelBuf.Subgraphs(0)
                    .Tensors(tensorIndex)
                    .Quantization()
                    .ZeroPoint(0),
                ),
                "quantization_parameters": {
                    "scales": modelBuf.Subgraphs(0)
                    .Tensors(tensorIndex)
                    .Quantization()
                    .ScaleAsNumpy(),
                    "zero_points": modelBuf.Subgraphs(0)
                    .Tensors(tensorIndex)
                    .Quantization()
                    .ZeroPointAsNumpy(),
                    "quantized_dimension": modelBuf.Subgraphs(0)
                    .Tensors(tensorIndex)
                    .Quantization()
                    .QuantizedDimension(),
                },
                "sparsity_parameters": {
                    modelBuf.Subgraphs(0).Tensors(tensorIndex).Sparsity()
                },
            }
            inputsList.append(details)

        return inputsList

    def get_output_details(
            self, model_index: int = 0
    ) -> List[Dict[str, Any]]:
        """! Reads the output tensor details from the model.
        @param output_index  The index of output tensor to target.
        @param model_index The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return Tensor details, including the index, name, shape, data type, and quantization
        parameters.
        """

        # Select correct model from models list
        model = self.get_model(model_index)
        modelBuf = Model.GetRootAsModel(model.model_content, 0)

        outputsList = []
        for output_ in range(0, modelBuf.Subgraphs(0).OutputsLength()):

            # Output tensor is last tensor
            tensorIndex = modelBuf.Subgraphs(0).Outputs(output_)

            dtype: Union[Type[Any]]
            # Generate dictionary of tensor details
            if modelBuf.Subgraphs(0).Tensors(tensorIndex).Type() == TensorType.INT8:
                dtype = np.int8
            elif modelBuf.Subgraphs(0).Tensors(tensorIndex).Type() == TensorType.INT32:
                dtype = np.int32
            elif modelBuf.Subgraphs(0).Tensors(tensorIndex).Type() == TensorType.FLOAT32:
                dtype = np.float32

            details = {
                "name": str(modelBuf.Subgraphs(0).Tensors(tensorIndex).Name())[
                    1:
                ].strip("'"),
                "index": tensorIndex,
                "shape": modelBuf.Subgraphs(0).Tensors(tensorIndex).ShapeAsNumpy(),
                "shape_signature": modelBuf.Subgraphs(0)
                .Tensors(tensorIndex)
                .ShapeSignatureAsNumpy(),
                "dtype": dtype,
                "quantization": (
                    modelBuf.Subgraphs(0).Tensors(tensorIndex).Quantization().Scale(0),
                    modelBuf.Subgraphs(0)
                    .Tensors(tensorIndex)
                    .Quantization()
                    .ZeroPoint(0),
                ),
                "quantization_parameters": {
                    "scales": modelBuf.Subgraphs(0)
                    .Tensors(tensorIndex)
                    .Quantization()
                    .ScaleAsNumpy(),
                    "zero_points": modelBuf.Subgraphs(0)
                    .Tensors(tensorIndex)
                    .Quantization()
                    .ZeroPointAsNumpy(),
                    "quantized_dimension": modelBuf.Subgraphs(0)
                    .Tensors(tensorIndex)
                    .Quantization()
                    .QuantizedDimension(),
                },
                "sparsity_parameters": {
                    modelBuf.Subgraphs(0).Tensors(tensorIndex).Sparsity()
                },
            }
            outputsList.append(details)

        return outputsList

    def set_model(
        self,
        model_path: Optional[str] = None,
        model_content: Optional[bytes] = None,
        params_path: Optional[str] = None,
        params_content: Optional[bytes] = None,
        model_index: int = 0,
        secondary_memory: bool = False,
        flash: bool = False,
    ) -> None:
        """! Adds a model to the interpreter's list of models.
        @param model_path The path to the model file (.tflite), alternative to model_content.
        @param model_content The byte array representing a model, alternative to model_path.
        @param params_path The path to the params file for the model,
        alternative to params_content (optional).
        @param params_content The byte array representing the model parameters,
        alternative to params_path (optional).
        @param model_index The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        """

        # Check model_path or model_content is valid
        if type(model_path) == str or model_content is not None:
            tile_found = False
            # Find correct model and replace
            for model in self.models:
                if model.tile == model_index:
                    model = self.modelData(
                        model_path,
                        model_content,
                        params_path,
                        params_content,
                        model_index,
                        secondary_memory,
                        flash,
                    )
                    tile_found = True
                    break
            # If model wasn't previously set, add it to list
            if not tile_found:
                self.models.append(
                    self.modelData(
                        model_path,
                        model_content,
                        params_path,
                        params_content,
                        model_index,
                        secondary_memory,
                        flash,
                    )
                )
            self.initialise_interpreter(model_index)

    def get_model(self, model_index: int = 0):
        for model in self.models:
            if model.tile == model_index:
                return model

    class modelData:
        """! The model data class
        A class that holds a model and data associated with a single model.
        """

        def __init__(
            self,
            model_path: Optional[str],
            model_content: Optional[bytes],
            params_path: Optional[str],
            params_content: Optional[bytes],
            model_index: int,
            secondary_memory: bool,
            flash: bool,
        ):
            """! Model data initializer.
            Sets up variables, generates a list of operators used in the model,
            and reads model and params paths into byte arrays (content).
            @param model_path Path to the model file (.tflite).
            @param model_content Model model_content (byte array).
            @param params_path Path to model parameters file.
            @param params_content Model parameters content (byte array)
            @param model_index The model to target, for interpreters that support multiple models
            running concurrently. Defaults to 0 for use with a single model.
            """
            self.model_path: Optional[str] = model_path
            self.model_content: Optional[bytes] = model_content
            self.params_path: Optional[str] = params_path
            self.params_content: Optional[bytes] = params_content
            self.tile: int = model_index
            self.secondary_memory = secondary_memory
            self.flash = flash
            self.opList: List[str] = []
            self.pathToContent()
            self.modelToOpList()

        def modelToOpList(self) -> None:
            """! Generates operator list from model."""

            # Load model
            buffer = self.model_content
            model = Model.GetRootAsModel(buffer, 0)
            self.opList = []

            # Iterate through operators in model and add operators to opList
            for y in range(0, model.Subgraphs(0).OperatorsLength()):
                opcode = model.OperatorCodes(
                    model.Subgraphs(0).Operators(y).OpcodeIndex()
                )
                # If custom opcode parse string
                if opcode.BuiltinCode() == 32:
                    self.opList.append(str(opcode.CustomCode()).strip("b'"))
                # If built in op code, decode
                else:
                    self.opList.append(opcode2name(opcode.BuiltinCode()))

        def pathToContent(self) -> None:
            """! Reads model and params paths to content (byte arrays)"""

            # Check if path exists but not content
            if self.model_content is None and self.model_path is not None:
                with open(self.model_path, "rb") as input_fd:
                    self.model_content = input_fd.read()

            # Check if params_path exits but not params_content
            if self.params_content is None and self.params_path is not None:
                with open(self.params_path, "rb") as input_fd2:
                    self.params_content = input_fd2.read()
            # If no params, set to empty byte array
            else:
                self.params_content = bytes([])
