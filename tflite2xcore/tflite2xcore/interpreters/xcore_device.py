# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import signal
import socket
import time
import json
import ctypes
import subprocess
import tempfile
import re

import portalocker  # type: ignore
import numpy as np
from pathlib import Path

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


def get_child_xgdb_proc(port):
    def run(cmd, stdin=b""):
        process = subprocess.Popen(
            cmd.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output, err = process.communicate(stdin)
        rc = process.returncode
        assert rc == 0, f"Error running cmd: {cmd}\n output: {err}"
        return output.decode("utf-8")

    ps_out = run("ps a")
    for line in ps_out.splitlines():
        xgdb_index = line.find("xgdb", 0)
        if xgdb_index > 0:
            pid = int(line.split()[0])
            cmds_file = line[xgdb_index:].split()[11]
            with open(cmds_file, "r") as fd:
                xgdb_session = fd.read().replace("\n", "")
                port_match = re.match(r".+localhost:(\d+).+", xgdb_session)
                if port_match:
                    xgdb_port = int(port_match.group(1))
                    if xgdb_port == port:
                        # print(
                        #     f"Found xgdb instance with PID: {pid} on port: {xgdb_port}"
                        # )
                        return pid


def run_test_model(xtag_id):
    port = get_open_port()

    __PARENT_DIR = Path(__file__).parent.absolute()
    test_model_exe = str(
        __PARENT_DIR
        / ".."
        / ".."
        / ".."
        / "utils"
        / "test_model"
        / "bin"
        / "test_model_xscope.xe"
    )

    cmd = [
        "xrun",
        "--xscope",
        "--xscope-port",
        f"localhost:{port}",
        "--id",
        f"{xtag_id}",
        test_model_exe,
    ]

    xrun_proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False
    )

    if xrun_proc:
        # wait for port to be opened
        while test_port_is_open(port):
            time.sleep(1)

        xrun_xgdb_child_pid = get_child_xgdb_proc(port)

    return xrun_proc.pid, xrun_xgdb_child_pid


class XCOREDeviceEndpoint(object):
    RECV_AWK_PROBE_ID = 0
    GET_TENSOR_PROBE_ID = 1

    def __init__(self):
        tool_path = os.environ.get("XMOS_TOOL_PATH")
        lib_path = os.path.join(tool_path, "lib", "xscope_endpoint.so")
        self.lib_xscope = ctypes.CDLL(lib_path)

        # create callbacks
        self._print_cb = self._print_callback_func()
        self.lib_xscope.xscope_ep_set_print_cb(self._print_cb)

        self._record_cb = self._record_callback_func()
        self.lib_xscope.xscope_ep_set_record_cb(self._record_cb)

        self.clear()

    def _print_callback_func(self):
        def func(timestamp, length, data):
            self.on_print(timestamp, data[0:length])

        return PRINT_CALLBACK(func)

    def _record_callback_func(self):
        def func(id_, timestamp, length, data_val, data_bytes):
            self.on_probe(id_, timestamp, length, data_val, data_bytes)

        return RECORD_CALLBACK(func)

    def _send_blob(self, blob):
        CHUCK_SIZE = 256
        for i in range(0, len(blob), CHUCK_SIZE):
            self._publish_blob_chunk_ready = False
            self.publish(blob[i : i + CHUCK_SIZE])
            # wait for RECV_AWK probe
            while not self._publish_blob_chunk_ready:
                pass

    def clear(self):
        self._get_tensor_ready = False
        self._publish_blob_chunk_ready = False
        self._get_tensor_buffer = None

    def on_print(self, timestamp, data):
        msg = data.decode("utf-8").rstrip()
        print(msg)

    def on_probe(self, id_, timestamp, length, data_val, data_bytes):
        if id_ == XCOREDeviceEndpoint.RECV_AWK_PROBE_ID:
            self._publish_blob_chunk_ready = True
        elif id_ == XCOREDeviceEndpoint.GET_TENSOR_PROBE_ID:
            self._get_tensor_buffer = data_bytes[0:length]
            self._get_tensor_ready = True

    def connect(self, hostname="localhost", port="10234"):
        return self.lib_xscope.xscope_ep_connect(hostname.encode(), port.encode())

    def disconnect(self):
        self.lib_xscope.xscope_ep_disconnect()

    def publish(self, data):

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
        self._get_tensor_ready = False
        self.publish(f"GET_TENSOR {index}\0".encode())
        # wait for reply
        while not self._get_tensor_ready:
            pass

        return self._get_tensor_buffer


class XCOREDeviceServer(object):
    @staticmethod
    def acquire():
        lock_path = Path("xcore_devices.lock")
        devices_path = Path("xcore_devices.json")
        with portalocker.Lock(lock_path, timeout=10) as fh:
            if not devices_path.is_file():
                p = subprocess.Popen(
                    ["xrun", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                output, err = p.communicate()
                if p.returncode == 0:
                    devices = []
                    lines = output.decode("utf-8").split("\n")
                    for line in lines[6:]:
                        if line.strip():
                            fields = line.strip().split()
                            devices.append(
                                {"id": fields[0], "adaptor": fields[-1],}
                            )
                    with open(devices_path, "w") as fd:
                        fd.write(json.dumps(devices))
                else:
                    err_str = err.decode("utf-8").strip()
                    raise Exception(f"Error {err_str}")

            with open(devices_path, "r+") as fd:
                devices = json.loads(fd.read())

                device = devices.pop()
                # TODO: if no xrun info in device dict, launch it & add it
                print()
                print()
                print()
                print(device)
                if "xrun_pid" not in device:
                    xrun_pid, xgdb_pid = run_test_model(device["id"])

                # TODO: create endpoint and return
                print(device)
                print()
                print()
                print()
                if devices:
                    fd.write(json.dumps(devices))

                return None

