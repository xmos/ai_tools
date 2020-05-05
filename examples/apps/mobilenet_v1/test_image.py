#!/usr/bin/env python

# Copyright (c) 2019, XMOS Ltd, All rights reserved
import sys
import os
import time
import ctypes

CHUCK_SIZE = 128

PRINT_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_char_p
)


class Endpoint(object):
    def __init__(self):
        tool_path = os.environ.get("XMOS_TOOL_PATH")
        lib_path = os.path.join(tool_path, "lib", "xscope_endpoint.so")
        self.lib_xscope = ctypes.CDLL(lib_path)

        self.ready = True
        self.lines = []

        self._print_cb = self._print_callback_func()
        self.lib_xscope.xscope_ep_set_print_cb(self._print_cb)

    def _print_callback_func(self):
        def func(timestamp, length, data):
            self.on_print(timestamp, data[0:length])

        return PRINT_CALLBACK(func)

    def on_print(self, timestamp, data):
        msg = data.decode().rstrip()

        if msg == "DONE!":
            self.ready = True
        else:
            self.lines.append(msg)

    def connect(self, hostname="localhost", port="10234"):
        return self.lib_xscope.xscope_ep_connect(hostname.encode(), port.encode())

    def disconnect(self):
        self.lib_xscope.xscope_ep_disconnect()

    def publish(self, data):
        self.ready = False

        return self.lib_xscope.xscope_ep_request_upload(
            ctypes.c_uint(len(data) + 1), ctypes.c_char_p(data)
        )


ep = Endpoint()

try:
    if ep.connect():
        print("Failed to connect")
    else:
        # time.sleep(5)
        with open(sys.argv[1], "rb") as fd:
            bits = fd.read()
            for i in range(0, len(bits), CHUCK_SIZE):
                retval = ep.publish(bits[i : i + CHUCK_SIZE])
        while not ep.ready:
            pass

except KeyboardInterrupt:
    pass

ep.disconnect()
print("\n".join(ep.lines))

