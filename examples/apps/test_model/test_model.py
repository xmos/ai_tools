#!/usr/bin/env python

# Copyright (c) 2020, XMOS Ltd, All rights reserved
import sys
import os
import ctypes

PRINT_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_char_p
)


class TestingXCoreInterpreterEndpoint(object):
    def __init__(self):
        tool_path = os.environ.get("XMOS_TOOL_PATH")
        lib_path = os.path.join(tool_path, "lib", "xscope_endpoint.so")
        self.lib_xscope = ctypes.CDLL(lib_path)

        self._print_cb = self._print_callback_func()
        self.lib_xscope.xscope_ep_set_print_cb(self._print_cb)

        self.clear()

    def _print_callback_func(self):
        def func(timestamp, length, data):
            self.on_print(timestamp, data[0:length])

        return PRINT_CALLBACK(func)

    def _send_blob(self, blob):
        CHUCK_SIZE = 128
        for i in range(0, len(blob), CHUCK_SIZE):
            self.publish(blob[i : i + CHUCK_SIZE])

    def clear(self):
        self.ready = True
        self.log = []
        self.output = []

    def on_print(self, timestamp, data):
        msg = data.decode().rstrip()

        if msg == "DONE!":
            self.ready = True
        elif msg.startswith("OUTPUT"):
            fields = msg.split(",")  # format is OUTPUT,integer index,1 byte hex value
            self.output.append(fields[2])
        else:
            self.log.append(msg)  # anything not prefixed with OUTPUT is a log message

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
                    print(" ".join(ep.output))
                    ep.clear()

except KeyboardInterrupt:
    pass

ep.disconnect()
print("\n".join(ep.log))
