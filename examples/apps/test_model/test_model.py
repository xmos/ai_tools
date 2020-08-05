#!/usr/bin/env python

# Copyright (c) 2020, XMOS Ltd, All rights reserved
import sys
import os
import ctypes

import numpy as np

PRINT_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_char_p
)

RECORD_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint,  # id
    ctypes.c_ulonglong,  # timestamp
    ctypes.c_uint,  # length
    ctypes.c_ulonglong,  # dataval
    ctypes.c_char_p,  # databytes
)

REGISTER_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint,  # id
    ctypes.c_uint,  # type
    ctypes.c_uint,  # r
    ctypes.c_uint,  # g
    ctypes.c_uint,  # b
    ctypes.c_char_p,  # name
    ctypes.c_char_p,  # unit
    ctypes.c_uint,  # data_type
    ctypes.c_char_p,  # data_name
)


class TestingXCoreInterpreterEndpoint(object):
    def __init__(self):
        tool_path = os.environ.get("XMOS_TOOL_PATH")
        lib_path = os.path.join(tool_path, "lib", "xscope_endpoint.so")
        self.lib_xscope = ctypes.CDLL(lib_path)

        # create callbacks
        self._print_cb = self._print_callback_func()
        self.lib_xscope.xscope_ep_set_print_cb(self._print_cb)

        self._record_cb = self._record_callback_func()
        self.lib_xscope.xscope_ep_set_record_cb(self._record_cb)

        self._register_cb = self._register_callback_func()
        self.lib_xscope.xscope_ep_set_register_cb(self._register_cb)

        self._probe_info = {}  # probe id to probe info lookup.
        # probe_info includes name, units, data type, etc...

        self.clear()

    def _print_callback_func(self):
        def func(timestamp, length, data):
            self.on_print(timestamp, data[0:length])

        return PRINT_CALLBACK(func)

    def _record_callback_func(self):
        def func(id_, timestamp, length, data_val, data_bytes):
            self.on_probe(id_, timestamp, length, data_val, data_bytes)

        return RECORD_CALLBACK(func)

    def _register_callback_func(self):
        def func(id_, type_, r, g, b, name, unit, data_type, data_name):
            self._probe_info[id_] = {
                "type": type_,
                "name": name.decode("utf-8"),
                "unit": unit.decode("utf-8"),
                "data_type": data_type,
            }

        return REGISTER_CALLBACK(func)

    def _send_blob(self, blob):
        CHUCK_SIZE = 128
        for i in range(0, len(blob), CHUCK_SIZE):
            self.publish(blob[i : i + CHUCK_SIZE])

    def clear(self):
        self.ready = True
        self.log = []
        self.output = None

    def on_print(self, timestamp, data):
        msg = data.decode().rstrip()

        self.log.append(msg)  # anything not prefixed with OUTPUT is a log message

    def on_probe(self, id_, timestamp, length, data_val, data_bytes):
        probe = self._probe_info[id_]
        if probe["name"] == "output_tensor":
            # print("length=", length)
            # print("data_bytes=", data_bytes.strip())
            self.output = np.frombuffer(data_bytes.strip(), dtype=np.int8)

            self.ready = True

    def connect(self, hostname="localhost", port="10234"):
        return self.lib_xscope.xscope_ep_connect(hostname.encode(), port.encode())

    def disconnect(self):
        self.lib_xscope.xscope_ep_disconnect()

    def publish(self, data):
        self.ready = False

        if (
            self.lib_xscope.xscope_ep_request_upload(
                ctypes.c_uint(len(data) + 1), ctypes.c_char_p(data)
            )
            != 0
        ):
            raise Exception("Error publishing data")

    def publish_model(self, model_content):
        self.publish(b"START_MODEL")
        self._send_blob(model_content)
        self.publish(b"END_MODEL")

    def publish_input(self, input_content):
        self._send_blob(input_content)


ep = TestingXCoreInterpreterEndpoint()

model_filename = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

try:
    if ep.connect():
        print("Failed to connect")
    else:
        print("Connected")

        # Send model
        for i in range(2):
            with open(model_filename, "rb") as fd:
                model_content = fd.read()
                ep.publish_model(model_content)

            # Send input
            with open(input_filename, "rb") as fd:
                input_tensor = fd.read()
                for i in range(2):
                    ep.publish_input(input_tensor)
                    # Wait for output
                    while not ep.ready:
                        pass
                    # Do something with the output
                    print(ep.output)
                    ep.clear()

except KeyboardInterrupt:
    pass

ep.disconnect()
print("\n".join(ep.log))
