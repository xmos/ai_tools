# Copyright (c) 2020, XMOS Ltd, All rights reserved

import usb
from typing import Tuple
import numpy as np

IOSERVER_INVOKE = int(0x01)
IOSERVER_TENSOR_SEND_OUTPUT = int(0x02)
IOSERVER_TENSOR_RECV_INPUT = int(0x03)
IOSERVER_RESET = int(0x07)
IOSERVER_EXIT = int(0x08)


class IOServerError(Exception):
    """Error from device"""

    pass


class IOError(IOServerError):
    """IO Error from device"""

    pass


def handle_usb_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except usb.core.USBError as e:
            print(f"USB error {e}")
            if e.backend_error_code == usb.backend.libusb1.LIBUSB_ERROR_PIPE:
                raise IOError()
            else:
                raise IOServerError(f"Wow...") from e

    return wrapper


class IOServer:
    def __init__(self, output_details: Tuple[dict, ...] = None, timeout=5000):
        self.__out_ep = None
        self.__in_ep = None
        self._dev = None
        self._output_details = output_details
        self._timeout = timeout
        self._max_block_size = 512  # TODO read from (usb) device?
        super().__init__()

    def bytes_to_arr(self, data_bytes, tensor_num):
        if self._output_details:
            d = self._output_details[tensor_num]
            s = d["shape"]
            return np.frombuffer(data_bytes, dtype=d["dtype"])[: np.prod(s)].reshape(s)
        return np.frombuffer(data_bytes, dtype=np.uint8)

    def write_input_tensor(self, raw_img, tensor_num=0, model_num=0):
        self._download_data(
            IOSERVER_TENSOR_RECV_INPUT,
            raw_img,
            tensor_num=tensor_num,
            model_num=model_num,
        )

    def read_output_tensor(self, tensor_num=0, model_num=0):
        # Retrieve result from device
        data_read = self._upload_data(
            IOSERVER_TENSOR_SEND_OUTPUT,
            model_num=model_num,
            tensor_num=tensor_num,
        )
        assert type(data_read) is bytearray
        return self.bytes_to_arr(data_read, tensor_num)

    def close(self):
        if self._dev is not None:
            self._dev.write(self._out_ep, bytes([IOSERVER_EXIT, 0, 0]), 1000)
            usb.util.dispose_resources(self._dev)
            self._dev = None

    @handle_usb_error
    def _download_data(self, cmd, data_bytes, tensor_num=0, model_num=0):
        # TODO rm this extra CMD packet
        self._out_ep.write(bytes([cmd, model_num, tensor_num]))
        self._out_ep.write(data_bytes, 1000)
        if (len(data_bytes) % self._max_block_size) == 0:
            self._out_ep.write(bytearray(), 1000)

    @handle_usb_error
    def _upload_data(self, cmd, tensor_num=0, model_num=0):
        read_data = bytearray()
        self._out_ep.write(bytes([cmd, model_num, tensor_num]), self._timeout)
        buff = usb.util.create_buffer(self._max_block_size)
        read_len = self._dev.read(self._in_ep, buff, 10000)
        read_data.extend(buff[:read_len])
        while read_len == self._max_block_size:
            read_len = self._dev.read(self._in_ep, buff, 10000)
            read_data.extend(buff[:read_len])

        return read_data

    def _clear_error(self):
        self._dev.clear_halt(self._out_ep)
        self._dev.clear_halt(self._in_ep)

    def connect(self):
        self._dev = None
        while self._dev is None:
            # TODO - more checks that we have the right device..
            self._dev = usb.core.find(idVendor=0x20B1, product="xAISRV")

        # set the active configuration. With no arguments, the first
        # configuration will be the active one
        self._dev.set_configuration()

        # get an endpoint instance
        cfg = self._dev.get_active_configuration()
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

        print("Connected to XCORE_IO_SERVER via USB")

    # TODO move to super()
    def start_inference(self):
        # Send cmd
        self._out_ep.write(bytes([IOSERVER_INVOKE, 0, 0]), 1000)

        # Send out a 0 length packet
        self._out_ep.write(bytes([]), 1000)

    def reset(self):
        # Send cmd
        self._out_ep.write(bytes([IOSERVER_RESET, 0, 0]), 1000)

        # Send out a 0 length packet
        self._out_ep.write(bytes([]), 1000)
