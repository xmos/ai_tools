# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import os
import signal
import socket
import time
import json
import ctypes
import subprocess
import atexit
import re
import logging

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
    """
    Get an open port number
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def test_port_is_open(port):
    """
    Check if a port is open
    """
    port_open = True
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", port))
    except OSError:
        port_open = False
    s.close()
    return port_open


def get_child_xgdb_pid(port):
    """
    Get the process ID of xrun's child xgdb process given the xrun port
    """

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
                        logging.debug(
                            f"Found xgdb instance with pid={pid} on port={xgdb_port}"
                        )
                        return pid


def run_test_model(xtag_id):
    """
    Run the test_model firmware
    """
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

        xrun_xgdb_child_pid = get_child_xgdb_pid(port)

    return xrun_proc.pid, xrun_xgdb_child_pid, port


def get_devices():
    """
    Get an array of all connected xcore devices
    """
    p = subprocess.Popen(
        ["xrun", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    output, err = p.communicate()
    if p.returncode == 0:
        devices = []
        lines = output.decode("utf-8").split("\n")
        for line in lines[6:]:
            if line.strip():
                fields = line.strip().split("\t")
                devices.append(
                    {
                        "id": fields[0].strip(),
                        "name": fields[1].strip(),
                        "adaptor_id": fields[2].strip(),
                        "devices": fields[3].strip(),
                    }
                )
    else:
        err_str = err.decode("utf-8").strip()
        raise Exception(f"Error {err_str}")

    return devices


class XCOREDeviceEndpoint(object):
    PING_AWK_PROBE_ID = 0
    RECV_AWK_PROBE_ID = 1
    GET_TENSOR_PROBE_ID = 2

    def __init__(self):
        tool_path = os.environ.get("XMOS_TOOL_PATH")
        lib_path = os.path.join(tool_path, "lib", "xscope_endpoint.so")
        self.lib_xscope = ctypes.CDLL(lib_path)

        # create callbacks
        self._print_cb = self._print_callback_func()
        self.lib_xscope.xscope_ep_set_print_cb(self._print_cb)

        self._record_cb = self._record_callback_func()
        self.lib_xscope.xscope_ep_set_record_cb(self._record_cb)

        self._hostname = None
        self._port = None

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
        CHUCK_SIZE = 128
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

    @property
    def port(self):
        return self._port

    def on_print(self, timestamp, data):
        msg = data.decode("utf-8").rstrip()
        logging.debug(msg)

    def on_probe(self, id_, timestamp, length, data_val, data_bytes):
        if id_ == XCOREDeviceEndpoint.PING_AWK_PROBE_ID:
            self._device_ready = True
        elif id_ == XCOREDeviceEndpoint.RECV_AWK_PROBE_ID:
            self._publish_blob_chunk_ready = True
        elif id_ == XCOREDeviceEndpoint.GET_TENSOR_PROBE_ID:
            self._get_tensor_buffer = data_bytes[0:length]
            self._get_tensor_ready = True

    def connect(self, hostname="localhost", port=10234):
        ep_connect = self.lib_xscope.xscope_ep_connect(
            hostname.encode(), str(port).encode()
        )
        self._hostname = hostname
        self._port = port
        return ep_connect

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

    def ping_device(self, timeout=5):
        SLEEP_DURATION = 0.2
        duration = 0
        self._device_ready = False

        self.publish(b"PING_RECV")
        # wait for PING_AWK probe
        while not self._device_ready:
            if duration >= timeout:
                break
            time.sleep(SLEEP_DURATION)
            duration += SLEEP_DURATION

        return self._device_ready

    def set_model(self, model_content):
        self.publish(b"START_MODEL\0")
        self._send_blob(model_content)
        self.publish(b"END_MODEL\0")

    def set_invoke(self):
        self.publish(b"CALL_INVOKE")

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
    FILE_LOCK_TIMEOUT = 20
    lock_path = Path("xcore_devices.lock")
    devices_path = Path("xcore_devices.json")

    @staticmethod
    def _setup_device_use(device):
        # setup device which means launcing xrun and
        #   and setting all pids. ports and in_use
        logging.debug(f"Setting up device: {device}")
        xrun_pid, xgdb_pid, xscope_port = run_test_model(device["id"])
        device["xrun_pid"] = xrun_pid
        device["xgdb_pid"] = xgdb_pid
        device["xscope_port"] = xscope_port
        device["parent_pid"] = os.getpid()
        device["in_use"] = False
        logging.debug(f"Device setup: {device}")

        return device

    @staticmethod
    def _reset_device_use(device):
        # reset device which means kill the xrun and xgdb processes
        #   and setting all pids. ports and in_use to None (or False)
        logging.debug(f"Resetting device: {device}")
        os.kill(device["xrun_pid"], signal.SIGKILL)
        os.kill(device["xgdb_pid"], signal.SIGKILL)

        device["xrun_pid"] = None
        device["xgdb_pid"] = None
        device["xscope_port"] = None
        device["parent_pid"] = None
        device["in_use"] = False
        logging.debug(f"Device reset: {device}")

        return device

    @staticmethod
    def _atexit():
        with portalocker.TemporaryFileLock(
            XCOREDeviceServer.lock_path, timeout=XCOREDeviceServer.FILE_LOCK_TIMEOUT
        ) as fh:
            if XCOREDeviceServer.devices_path.is_file():
                # search (by parent pid) for this process in cached devices
                with open(XCOREDeviceServer.devices_path, "r+") as fd:
                    this_pid = os.getpid()
                    cached_devices = json.loads(fd.read())
                    for cached_device in cached_devices:
                        if this_pid == cached_device["parent_pid"]:
                            cached_device = XCOREDeviceServer._reset_device_use(
                                cached_device
                            )

                with open(XCOREDeviceServer.devices_path, "w") as fd:
                    fd.write(json.dumps(cached_devices))

    @staticmethod
    def acquire():
        atexit.register(XCOREDeviceServer._atexit)
        with portalocker.TemporaryFileLock(
            XCOREDeviceServer.lock_path, timeout=XCOREDeviceServer.FILE_LOCK_TIMEOUT
        ):
            # get the connected devices
            connected_devices = get_devices()
            # get the cached devices
            if XCOREDeviceServer.devices_path.is_file():
                with open(XCOREDeviceServer.devices_path, "r+") as fd:
                    cached_devices = json.loads(fd.read())
            else:
                cached_devices = []

            logging.debug(f"Connected devices: {connected_devices}")
            logging.debug(f"Cached devices: {cached_devices}")
            # sync connected and cached devices
            synced_devices = []
            for connected_device in connected_devices:
                # check cached_devices for this connected device
                new_device = True  # until proven otherwise
                for cached_device in cached_devices:
                    if cached_device["adaptor_id"] == connected_device["adaptor_id"]:
                        logging.debug(f"Found cached device: {cached_device}")
                        if cached_device.get("xrun_pid", None) == None:
                            # cached device but need to be setup
                            cached_device = XCOREDeviceServer._setup_device_use(
                                cached_device
                            )
                        else:
                            # ensure device is responding
                            ep = XCOREDeviceEndpoint()
                            port = cached_device["xscope_port"]
                            ep.connect(port=port)
                            logging.debug(f"Pinging port: {port}")
                            if ep.ping_device():
                                logging.debug("Ping succeeded")
                            else:
                                logging.debug("Ping failed")
                                # device did not respond so reset it
                                cached_device = XCOREDeviceServer._reset_device_use(
                                    cached_device
                                )
                                # then setup the device
                                cached_device = XCOREDeviceServer._setup_device_use(
                                    cached_device
                                )
                            ep.disconnect()

                        # this is a known device so save in synced devices
                        synced_devices.append(cached_device)
                        new_device = False
                        break

                if new_device:
                    # connected device not found in cached devices so it must be new
                    logging.debug(f"Found new device: {connected_device}")
                    # setup this new device
                    connected_device = XCOREDeviceServer._setup_device_use(
                        connected_device
                    )
                    # new devices are save in synced devices
                    synced_devices.append(connected_device)

            # now search synced_devices for an available device
            acquired_device = None
            for synced_device in synced_devices:
                if synced_device.get("in_use", False) == False:
                    acquired_device = synced_device
                    acquired_device["in_use"] = True
                    break

            # save the synched devices
            logging.debug(f"Saving synced devices: {synced_devices}")
            with open(XCOREDeviceServer.devices_path, "w") as fd:
                fd.write(json.dumps(synced_devices))

            logging.debug(f"Acquired device: {acquired_device}")

            # create endpoint from acquired_device and return
            ep = XCOREDeviceEndpoint()
            ep.connect(port=acquired_device["xscope_port"])

            return ep

    @staticmethod
    def release(endpoint):
        with portalocker.TemporaryFileLock(
            XCOREDeviceServer.lock_path, timeout=XCOREDeviceServer.FILE_LOCK_TIMEOUT
        ):
            if XCOREDeviceServer.devices_path.is_file():
                # search (by port) for endpoint in cached devices
                with open(XCOREDeviceServer.devices_path, "r+") as fd:
                    cached_devices = json.loads(fd.read())
                    for cached_device in cached_devices:
                        if endpoint.port == cached_device["xscope_port"]:
                            endpoint.disconnect()
                            cached_device["in_use"] = False
                            logging.debug(f"Released device: {cached_device}")

                with open(XCOREDeviceServer.devices_path, "w") as fd:
                    fd.write(json.dumps(cached_devices))
