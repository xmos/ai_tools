# Copyright 2022 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import sys
from abc import abstractmethod
from typing import List, Union, Tuple, Optional, Any, Dict

import numpy as np
from numpy import ndarray

import xmos_ai_tools.xinterpreters.device.aisrv_cmd as aisrv_cmd
from xmos_ai_tools.xinterpreters.base.base_interpreter import (
    xcore_tflm_base_interpreter,
)

XCORE_IE_MAX_BLOCK_SIZE = 512


class AISRVError(Exception):
    """Error from device"""

    pass


class IOError(AISRVError):
    """IO Error from device"""

    pass


class NoModelError(AISRVError):
    """No model error from device"""

    pass


class ModelError(AISRVError):
    """Model error from device"""

    pass


class InferenceError(AISRVError):
    """Inference Error from device"""

    pass


class CommandError(AISRVError):
    """Command Error from device"""

    pass


class xcore_tflm_device_interpreter(xcore_tflm_base_interpreter):
    """! The xcore interpreters device class.
    To be inherited by usb/spi interpreters, inherits from base interpreter.
    """

    def __init__(self):
        """! Device interpreter initializer.
        Calls the connect function to connect to a device over the current interface.
        """
        self._timings_length = None
        self._max_block_size = XCORE_IE_MAX_BLOCK_SIZE  # TODO read from (usb) device?
        self.connect()
        super().__init__()

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """! Exit calls close function to delete interpreter"""
        self.close()

    def initialise_interpreter(self, model_index: int = 0) -> None:
        """! Abstract initialising interpreter with model associated with model_index.
        @param model_index The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        """
        model = self.get_model(model_index)
        self.download_model(
            bytearray(model.model_content),
            model.secondary_memory,
            model.flash,
            model.tile,
        )
        return

    def set_tensor(self, tensor_index, value: ndarray, model_index=0) -> None:
        """! Abstract for writing the input tensor of a model.
        @param tensor_index  The index of input tensor to target.
        @param value  The blob of data to set the tensor to.
        @param model_index The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        """
        count: Optional[int]
        tensor_details: Optional[Dict[str, Any]]
        count, tensor_details = next(
            filter(lambda x: x[1]["index"] == tensor_index, enumerate(self.get_input_details())),
            (None, None)
        )

        if count is None or tensor_details is None:
            print(f"No tensor at index {tensor_index} found.", sys.stderr)
            raise IndexError

        tensor_type = tensor_details["dtype"]
        if tensor_type == np.int32 or tensor_type == np.float32:
            bpi = 4
        else:
            bpi = 1
        self._download_data(
            aisrv_cmd.CMD_SET_INPUT_TENSOR,
            value.tobytes(),
            tensor_num=count,
            engine_num=model_index,
        )
        print("Setting Input Tensor")
        return

    def get_tensor(self, tensor_index: int = 0, model_index: int = 0, tensor: ndarray = None) -> ndarray:
        """! Abstract for reading the data in the output tensor of a model.
        @param tensor_index  The index of output tensor to target.
        @param tensor Tensor of correct shape to write into (optional)
        @param model_index The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return The data that was stored in the output tensor.
        """

        count: Optional[int]
        tensor_details: Optional[Dict[str, Any]]
        count, tensor_details = next(
            filter(lambda x: x[1]["index"] == tensor_index, enumerate(self.get_output_details())),
            (None, None)
        )

        if count is None or tensor_details is None:
            print(f"No tensor at index {tensor_index} found.", sys.stderr)
            raise IndexError

        tensor_type = tensor_details["dtype"]
        tensor_length = self.get_tensor_size(tensor_index, model_index)

        data_read = self._upload_data(
            aisrv_cmd.CMD_GET_OUTPUT_TENSOR,
            tensor_length,
            tensor_num=count,
            engine_num=model_index,
        )

        assert type(data_read) == list
        assert type(data_read[0]) == int

        x = np.array(data_read, np.uint8)
        bytes = x.tobytes()
        output = np.frombuffer(bytes, dtype=tensor_type)

        return np.reshape(output, tensor_details["shape"])

    def get_input_tensor(self, input_index=0, model_index=0) -> List[Union[int, Tuple[float]]]:
        """! Abstract for reading the data in the input tensor of a model.
        @param input_index  The index of output tensor to target.
        @param model_index The engine to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        @return The data that was stored in the input tensor.
        """
        # Retrieve result from device
        input_length = self.get_input_tensor_size(input_index, model_index)
        data_read = self._upload_data(
            aisrv_cmd.CMD_GET_INPUT_TENSOR,
            input_length,
            tensor_num=input_index,
            engine_num=model_index,
        )

        assert type(data_read) == list
        assert type(data_read[0]) == int

        x = np.array(data_read, np.uint8)
        bytes = x.tobytes()
        output = np.frombuffer(bytes, dtype=np.uint32)

        return output.tolist()

    @abstractmethod
    def invoke(self, model_index=0) -> None:
        pass

    @abstractmethod
    def close(self, model_index=0) -> None:
        """! Abstract deleting the interpreter
        @params model_index Defines which interpreter to target in systems with multiple
        """
        return

    def tensor_arena_size(self) -> int:
        """! Abstract to read the size of the tensor arena required
        @return size of the tensor arena as an integer
        """
        raise NotImplementedError

    def _check_status(self, status):
        """! Abstract to read a status code and raise an exception
        @param status Status code
        """
        raise NotImplementedError

    def print_memory_plan(self) -> None:
        """! Abstract to print a plan of memory allocation"""
        raise NotImplementedError

    # Internal Device Interpreter Functions

    @abstractmethod
    def connect(self):
        """! Abstract to connect to a connected device"""
        pass

    @abstractmethod
    def _clear_error(self):
        """! Abstract to clear errors on the device"""
        pass

    def download_model(
            self, model_bytes: bytearray, secondary_memory: bool = False, flash: bool = False, model_index: int = 0
    ):
        """! Download a model on to the device.
        @param model_bytes  The byte array containing the model.
        @param secondary_memory  Download the model to primary and secondary memory.
        @param flash  Store the model in flash memory.
        @param model_index  The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        """

        if not flash:
            assert type(model_bytes) == bytearray

            print("Model length (bytes): " + str(len(model_bytes)))

            if secondary_memory:
                print("Downloading model to secondary memory")
                cmd = aisrv_cmd.CMD_SET_MODEL_SECONDARY
            else:
                print("Downloading model to primary memory")
                cmd = aisrv_cmd.CMD_SET_MODEL_PRIMARY

        elif flash:
            if secondary_memory:
                print("Loading model to secondary memory")
                cmd = aisrv_cmd.CMD_SET_MODEL_SECONDARY_FLASH
            else:
                print("Loading model to primary memory")
                cmd = aisrv_cmd.CMD_SET_MODEL_PRIMARY_FLASH

        try:
            # Download model to device
            self._download_data(cmd, model_bytes, engine_num=model_index)
        except IOError:
            print("IO Error\n")
            raise IOError

    def read_debug_log(self) -> str:
        """! Read the debug log on device (TFLM Error Reporter)."""
        debug_string = self._upload_data(
            aisrv_cmd.CMD_GET_DEBUG_LOG, 256
        )  # TODO rm magic number

        r = bytearray(debug_string).decode("utf8", errors="replace")
        return r

    def read_times(self, model_index: int = 0) -> List[Union[int, Tuple[float]]]:
        """! Read the operator timings from a completed inference.
        @param model_index  The model to target, for interpreters that support multiple models
        running concurrently. Defaults to 0 for use with a single model.
        """
        model = self.get_model(model_index)
        ops_length = len(model.opList)

        times_bytes = self._upload_data(
            aisrv_cmd.CMD_GET_TIMINGS, ops_length * 4, engine_num=model_index
        )

        x = np.array(times_bytes, np.uint8)
        bytes = x.tobytes()
        output = np.frombuffer(bytes, dtype=np.uint32)

        return output.tolist()

    @abstractmethod
    def _upload_data(self, cmd, length, sign=False, tensor_num=0, engine_num=0) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def _download_data(self, cmd, data_bytes, tensor_num=0, engine_num=0) -> bytes:
        raise NotImplementedError


class xcore_tflm_usb_interpreter(xcore_tflm_device_interpreter):
    def __init__(self, timeout=500000):
        self._out_ep = None
        self._in_ep = None
        self._dev = None
        self._timeout = timeout
        super().__init__()

    def _download_data(self, cmd, data_bytes, tensor_num=0, engine_num=0):
        import usb

        try:
            self._out_ep.write(bytes([cmd, engine_num, tensor_num]))
            self._out_ep.write(data_bytes, 1000)

            if (len(data_bytes) % self._max_block_size) == 0:
                self._out_ep.write(bytearray([]), 1000)

        except usb.core.USBError as e:
            if e.backend_error_code == usb.backend.libusb1.LIBUSB_ERROR_PIPE:
                print("USB error, DOWNLOAD IN/OUT pipe halted")
                self._clear_error()
                print("DEBUG LOG: ", self.read_debug_log())
                raise IOError()

    def _upload_data(self, cmd, length, sign=False, tensor_num=0, engine_num=0):
        import usb

        read_data = []

        try:
            self._out_ep.write(bytes([cmd, engine_num, tensor_num]), self._timeout)
            buff = usb.util.create_buffer(self._max_block_size)

            while True:
                read_len = self._dev.read(self._in_ep, buff, 10000)
                read_data.extend(buff[:read_len])
                if read_len != self._max_block_size:
                    break

            return read_data

        except usb.core.USBError as e:
            if e.backend_error_code == usb.backend.libusb1.LIBUSB_ERROR_PIPE:
                print("USB error, UPLOAD IN/OUT pipe halted")
                self._clear_error()
                print("DEBUG LOG: ", self.read_debug_log())
                raise IOError()

    def _clear_error(self):
        self._dev.clear_halt(self._out_ep)
        self._dev.clear_halt(self._in_ep)

    def connect(self):
        import usb

        self._dev = None
        while self._dev is None:
            # TODO - more checks that we have the right device..
            self._dev = usb.core.find(idVendor=0x20B1, idProduct=0xA15E)

            # set the active configuration. With no arguments, the first
            # configuration will be the active one
            self._dev.set_configuration()

            # get an endpoint instance
            cfg = self._dev.get_active_configuration()

            # print("found device: \n" + str(cfg))
            intf = cfg[(0, 0)]

            self._out_ep = usb.util.find_descriptor(
                intf,
                # match the first OUT endpoint
                custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress)
                                       == usb.util.ENDPOINT_OUT,
            )

            self._in_ep = usb.util.find_descriptor(
                intf,
                # match the first IN endpoint
                custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress)
                                       == usb.util.ENDPOINT_IN,
            )

            assert self._out_ep is not None
            assert self._in_ep is not None

            print("Connected to AISRV via USB")

    def invoke(self, model_index=0):
        # Send cmd
        print("Inferencing...")
        self._out_ep.write(bytes([aisrv_cmd.CMD_START_INFER, model_index, 0]), 1000)
        # Send out a 0 length packet
        self._out_ep.write(bytes([]), 1000)

    def close(self, model_index=0) -> None:
        import usb

        usb.util.dispose_resources(self._dev)
        return

    def start_acquire_single(self, sx, ex, sy, ey, rw, rh, engine_num=0):
        # Send cmd
        self._out_ep.write(
            bytes([aisrv_cmd.CMD_START_ACQUIRE_SINGLE, engine_num, 0]), 1000
        )

        def tobytes(l):
            o = []
            for i in l:
                o.append(i & 0xFF)
                o.append((i >> 8) & 0xFF)
            return bytes(o)

        # Send out packet with coordinates
        self._out_ep.write(tobytes([sx, ex, sy, ey, rw, rh]), 1000)

    def acquire_set_i2c(self, i2c_address, reg_address, reg_value, engine_num=0):
        # Send cmd
        self._out_ep.write(
            bytes([aisrv_cmd.CMD_START_ACQUIRE_SET_I2C, engine_num, 0]), 1000
        )

        def tobytes(l):
            o = []
            for i in l:
                o.append(i & 0xFF)
            return bytes(o)

        # Send out packet with coordinates
        self._out_ep.write(tobytes([i2c_address, reg_address, reg_value]), 1000)

    def start_acquire_stream(self, engine_num=0):
        # Send cmd
        self._out_ep.write(
            bytes([aisrv_cmd.CMD_START_ACQUIRE_STREAM, engine_num, 0]), 1000
        )

        # Send out a 0 length packet
        self._out_ep.write(bytes([]), 1000)

    def enable_output_gpio(self, engine_num=0):
        self._out_ep.write(
            bytes([aisrv_cmd.CMD_SET_OUTPUT_GPIO_EN, engine_num, 0]), 1000
        )
        self._out_ep.write(bytes([1]), 1000)

    def disable_output_gpio(self, engine_num=0):
        self._out_ep.write(
            bytes([aisrv_cmd.CMD_SET_OUTPUT_GPIO_EN, engine_num, 0]), 1000
        )
        self._out_ep.write(bytes([0]), 1000)

    def set_output_gpio_threshold(self, index, threshold):
        self._out_ep.write(
            bytes([aisrv_cmd.CMD_SET_OUTPUT_GPIO_THRESH, engine_num, 0]), 1000
        )
        self._out_ep.write(bytes([index, threshold]), 1000)

    def set_output_gpio_mode_max(self, engine_num=0):
        self._out_ep.write(
            bytes([aisrv_cmd.CMD_SET_OUTPUT_GPIO_MODE, engine_num, 0]), 1000
        )
        self._out_ep.write(bytes([1]), 1000)

    def set_output_gpio_mode_none(self, engine_num=0):
        self._out_ep.write(
            bytes([aisrv_cmd.CMD_SET_OUTPUT_GPIO_MODE, engine_num, 0]), 1000
        )
        self._out_ep.write(bytes([0]), 1000)
