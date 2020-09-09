# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import ctypes
import socket
import time
import subprocess

from tflite2xcore.xcore_interpreter import XCOREInterpreter

import numpy as np

MAX_DEVICE_MODEL_CONTENT_SIZE = 100000
MAX_DEVICE_TENSOR_ARENA_SIZE = 100000


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


def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def test_port_is_open(port):
    port_open = True
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", port))
    except OSError:
        port_open = False
    s.close()
    return port_open


class XCOREDeviceInterpreterEndpoint(object):
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
        self._get_tensor_buffer = None

    def on_print(self, timestamp, data):
        msg = data.decode("utf-8").rstrip()

        print(msg)

    def on_probe(self, id_, timestamp, length, data_val, data_bytes):
        probe = self._probe_info[id_]
        if probe["name"] == "get_tensor":
            self._get_tensor_buffer = data_bytes[0:length]
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

    def set_model(self, model_content):
        self.publish(b"START_MODEL\0")
        self._send_blob(model_content)
        self.publish(b"END_MODEL\0")

    def set_invoke(self):
        self.publish(b"INVOKE")

    def set_tensor(self, index, tensor_content):
        size = len(tensor_content)
        self.publish(f"SET_TENSOR {index} {size}\0".encode())
        self._send_blob(tensor_content)

    def get_tensor(self, index):
        self.publish(f"GET_TENSOR {index}\0".encode())
        # wait for reply
        while not self.ready:
            pass

        return self._get_tensor_buffer


class XCOREDeviceInterpreter(XCOREInterpreter):
    def __init__(
        self,
        model_path=None,
        model_content=None,
        max_tensor_arena_size=MAX_DEVICE_TENSOR_ARENA_SIZE,
    ):
        # verify model content size is not too large
        if len(model_content) > MAX_DEVICE_MODEL_CONTENT_SIZE:
            raise ValueError(f"model_content > {MAX_DEVICE_MODEL_CONTENT_SIZE} bytes")

        # verify max_tensor_arena_size is not too large
        if max_tensor_arena_size > MAX_DEVICE_TENSOR_ARENA_SIZE:
            raise ValueError(
                f"max_tensor_arena_size > {MAX_DEVICE_TENSOR_ARENA_SIZE} bytes"
            )

        super().__init__(model_path, model_content, max_tensor_arena_size)

        self._model_content = model_content

        self._xrun_proc = None

        self.start()

    def __del__(self):
        self.stop()

    def start(self, use_xsim=True):
        # start firmware
        port = get_open_port()
        # test_model_exe = "../../../../utils/test_model/bin/test_model.xe"
        test_model_exe = (
            "/home/kmoulton/repos/hotdog/ai_tools/utils/test_model/bin/test_model.xe"
        )

        use_xsim = True  # hard-coded for now
        if use_xsim:
            cmd = ["xsim", "--xscope", f"-realtime localhost:{port}", test_model_exe]
        else:
            # xtag_id = 0  # hard-coded for now
            cmd = [
                "xrun",
                "--io",
                "--xscope",
                "--xscope-port",
                f"localhost:{port}",
                # "--id",
                # xtag_id,
                test_model_exe,
            ]

        self._xrun_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False
        )

        if self._xrun_proc:
            # wait for port to be opened
            while test_port_is_open(port):
                time.sleep(0.1)

            # start xsope endpoint
            self._endpoint = XCOREDeviceInterpreterEndpoint()
            self._endpoint.connect(hostname="localhost", port=str(port))

    def stop(self):
        if self._xrun_proc:
            self._endpoint.disconnect()
            self._xrun_proc.terminate()
            # self._xrun_proc.kill()
            self._xrun_proc = None

    def allocate_tensors(self):
        super().allocate_tensors()

        # send model to device
        self._endpoint.set_model(self._model_content)

    def invoke(
        self,
        *,
        preinvoke_callback=None,
        postinvoke_callback=None,
        capture_op_states=False,
    ):

        if preinvoke_callback != None or postinvoke_callback != None:
            raise NotImplementedError("Callbacks not implemented")

        self._endpoint.set_invoke()

    def set_tensor(self, tensor_index, value):
        self._verify_allocated()
        self._endpoint.set_tensor(tensor_index, value.tobytes())

    def get_tensor(self, tensor_index):
        self._verify_allocated()

        tensor_details = self.get_tensor_details()[tensor_index]

        buffer = self._endpoint.get_tensor(tensor_index)
        self._endpoint.clear()

        tensor = np.frombuffer(buffer, tensor_details["dtype"])
        tensor = tensor.reshape(tensor_details["shape"])

        return tensor
