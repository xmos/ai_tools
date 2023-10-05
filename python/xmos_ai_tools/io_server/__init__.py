# Copyright (c) 2020, XMOS Ltd, All rights reserved
from abc import ABC, abstractmethod

import sys
import struct
import array

IOSERVER_INVOKE = int(0x01)
IOSERVER_TENSOR_SEND_OUTPUT = int(0x02)
IOSERVER_TENSOR_RECV_INPUT = int(0x03)


class XMOS_IO_SERVER(Exception):
    """Error from device"""

    pass


class IOError(XMOS_IO_SERVER):
    """IO Error from device"""

    pass


class InferenceError(XMOS_IO_SERVER):
    """Inference Error from device"""

    pass


class xmos_io_server(ABC):
    def __init__(self, timeout=5000):
        self.__out_ep = None
        self.__in_ep = None
        self._dev = None
        self._timeout = timeout
        self._max_block_size = 512  # TODO read from (usb) device?
        super().__init__()

    def bytes_to_ints(self, data_bytes, bpi=1):
        output_data_int = []

        # TODO better way of doing this?
        for i in range(0, len(data_bytes), bpi):
            x = data_bytes[i : i + bpi]
            y = int.from_bytes(x, byteorder="little", signed=True)
            output_data_int.append(y)

        return output_data_int

    def write_input_tensor(self, raw_img, tensor_num=0, model_num=0):
        self._download_data(
            IOSERVER_TENSOR_RECV_INPUT,
            raw_img,
            tensor_num=tensor_num,
            model_num=model_num,
        )

    def read_output_tensor(self, length, tensor_num=0, model_num=0):
        # Retrieve result from device
        data_read = self._upload_data(
            IOSERVER_TENSOR_SEND_OUTPUT,
            length,
            model_num=model_num,
            tensor_num=tensor_num,
        )

        assert type(data_read) == list

        return self.bytes_to_ints(data_read)

    def _download_data(self, cmd, data_bytes, tensor_num=0, model_num=0):
        import usb

        # print('Len ', len(data_bytes))
        try:
            # TODO rm this extra CMD packet
            self._out_ep.write(bytes([cmd, model_num, tensor_num]))

            # data_bytes = bytes([cmd]) + data_bytes
            # print('Written ', len(data_bytes))

            self._out_ep.write(data_bytes, 1000)
            # print('Done ', len(data_bytes))

            if (len(data_bytes) % self._max_block_size) == 0:
                self._out_ep.write(bytearray([]), 1000)

        except usb.core.USBError as e:
            print("USB error  ", str(e))
            if e.backend_error_code == usb.backend.libusb1.LIBUSB_ERROR_PIPE:
                raise IOError()

    def _upload_data(self, cmd, length, tensor_num=0, model_num=0):
        import usb

        read_data = []

        try:
            self._out_ep.write(bytes([cmd, model_num, tensor_num]), self._timeout)
            buff = usb.util.create_buffer(self._max_block_size)
            while True:
                read_len = self._dev.read(self._in_ep, buff, 10000)
                read_data.extend(buff[:read_len])
                # print('_Up got ', read_len)
                if read_len != self._max_block_size:
                    break

            return read_data

        except usb.core.USBError as e:
            if e.backend_error_code == usb.backend.libusb1.LIBUSB_ERROR_PIPE:
                # print("USB error, IN/OUT pipe halted")
                raise IOError()
            else:
                print("Wow ", e)

    def _clear_error(self):
        self._dev.clear_halt(self._out_ep)
        self._dev.clear_halt(self._in_ep)

    def connect(self):
        import usb

        # import usb.backend.libusb1

        # backend = usb.backend.libusb1.get_backend(find_library=lambda x: "/Users/deepakpanickal/Downloads/libusb-1.0.26-binaries/macos_11.6/lib/libusb-1.0.dylib")
        self._dev = None
        while self._dev is None:
            # TODO - more checks that we have the right device..
            self._dev = usb.core.find(idVendor=0x20B1)  # , idProduct=0xa15e)

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
