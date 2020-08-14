#!/usr/bin/env python

#
# Copyright (c) 2020, XMOS Ltd, All rights reserved
import sys
import os
import ctypes

import numpy as np
import tensorflow as tf

from tflite2xcore.xcore_interpreter import XCOREInterpreter
from tflite2xcore.model_generation.utils import quantize, dequantize

PRINT_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_char_p
)

RECORD_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint,  # id
    ctypes.c_ulonglong,  # timestamp
    ctypes.c_uint,  # length
    ctypes.c_ulonglong,  # dataval
    ctypes.POINTER(ctypes.c_char),  # data_bytes
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

        self.log.append(msg)

    def on_probe(self, id_, timestamp, length, data_val, data_bytes):
        probe = self._probe_info[id_]
        if probe["name"] == "output_tensor":
            self.output = np.frombuffer(data_bytes[0:length], dtype=np.int8)
            self.ready = True

    def connect(self, hostname="localhost", port="10234"):
        return self.lib_xscope.xscope_ep_connect(hostname.encode(), port.encode())

    def disconnect(self):
        self.lib_xscope.xscope_ep_disconnect()

    def publish(self, data):
        self.ready = False

        if (
            self.lib_xscope.xscope_ep_request_upload(
                ctypes.c_uint(len(data)), ctypes.c_char_p(data)
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


if __name__ == "__main__":
    xcore_model_filename = sys.argv[1]
    quant_model_filename = sys.argv[2]

    interpreter = XCOREInterpreter(model_path=xcore_model_filename)
    interpreter.allocate_tensors()

    # xcore model
    input_details = interpreter.get_input_details()[0]
    input_idx = input_details["index"]
    input_shape = input_details["shape"]
    input_quant = input_details["quantization"]
    output_quant = interpreter.get_output_details()[0]["quantization"]

    np.random.seed(42)
    x = np.random.uniform(-128, 128, size=input_shape).astype(np.int8)
    interpreter.set_tensor(input_idx, x)

    print("*************************")
    print(" Testing w/ interpreter  ")
    print("*************************")
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    output_idx = output_details["index"]

    y_xcore_interp = interpreter.get_tensor(output_idx)

    # reference quantized model
    interpreter_ref = tf.lite.Interpreter(model_path=quant_model_filename)
    interpreter_ref.allocate_tensors()

    input_details_ref = interpreter_ref.get_input_details()[0]
    input_idx_ref = input_details_ref["index"]

    x_ref = dequantize(x, *input_quant)
    interpreter_ref.set_tensor(input_idx_ref, x_ref)

    output_details_ref = interpreter_ref.get_output_details()[0]
    output_idx_ref = output_details_ref["index"]

    y_ref = interpreter_ref.get_tensor(output_idx_ref)
    y_ref_quantized = quantize(y_ref, *output_quant)
    print(y_ref_quantized - y_xcore_interp)

    print("*************************")
    print(" Testing w/ xsim or xrun")
    print("*************************")
    ep = TestingXCoreInterpreterEndpoint()
    ep.connect()
    # Send model
    with open(xcore_model_filename, "rb") as fd:
        model_content = fd.read()
        ep.publish_model(model_content)

    # Send input
    ep.publish_input(x.tobytes())
    # Wait for output
    while not ep.ready:
        pass
    y_xcore_hw = ep.output.reshape(y_xcore_interp.shape)
    print(y_ref_quantized - y_xcore_hw)

    print()
    print(ep.log)
    ep.clear()

    ep.disconnect()
