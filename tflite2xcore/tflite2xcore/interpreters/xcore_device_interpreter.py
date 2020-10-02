# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import numpy as np

from .xcore_interpreter import XCOREInterpreter
from .xcore_device import XCOREDeviceServer

MAX_DEVICE_MODEL_CONTENT_SIZE = 500000
MAX_DEVICE_TENSOR_ARENA_SIZE = 250000


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

        # self._xrun_proc = None
        # self._xrun_xgdb_child_pid = None

        self._device = None
        # self._set_model = False

        # self.start()

    # def __del__(self):
    #     self.stop()

    # def start(self, use_xsim=False):
    # start firmware
    # port = get_open_port()

    # __PARENT_DIR = Path(__file__).parent.absolute()
    # test_model_exe = str(
    #     __PARENT_DIR
    #     / ".."
    #     / ".."
    #     / ".."
    #     / "utils"
    #     / "test_model"
    #     / "bin"
    #     / "test_model_xscope.xe"
    # )

    # if use_xsim:
    #     cmd = ["xsim", "--xscope", f"-realtime localhost:{port}", test_model_exe]
    # else:
    #     xtag_id = 0  # hard-coded for now
    #     cmd = [
    #         "xrun",
    #         "--xscope",
    #         # "--xscope-realtime",
    #         "--xscope-port",
    #         f"localhost:{port}",
    #         "--id",
    #         f"{xtag_id}",
    #         test_model_exe,
    #     ]

    # self._xrun_proc = subprocess.Popen(
    #     cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False
    # )

    # if self._xrun_proc:
    #     # wait for port to be opened
    #     while test_port_is_open(port):
    #         time.sleep(1)

    #     if not use_xsim:
    #         self._xrun_xgdb_child_pid = get_child_xgdb_proc(port)

    #     # start xsope endpoint
    #     self._endpoint = XCOREDeviceInterpreterEndpoint()
    #     self._endpoint.connect(hostname="localhost", port=str(port))

    # def stop(self):
    # if self._xrun_proc:
    #     self._endpoint.disconnect()
    #     self._xrun_proc.terminate()
    #     if self._xrun_xgdb_child_pid:
    #         os.kill(self._xrun_xgdb_child_pid, signal.SIGTERM)
    #         self._xrun_xgdb_child_pid = None
    #     self._xrun_proc = None

    def allocate_tensors(self):
        super().allocate_tensors()

        if not self._device:
            self._device = XCOREDeviceServer.acquire()
            # send model to device
            # self._endpoint.set_model(self._model_content)

        # if not self._set_model:
        #     # send model to device
        #     self._endpoint.set_model(self._model_content)
        #     self._set_model = True

    def invoke(
        self,
        *,
        preinvoke_callback=None,
        postinvoke_callback=None,
        capture_op_states=False,
    ):

        if preinvoke_callback != None or postinvoke_callback != None:
            raise NotImplementedError("Callbacks not implemented")

        # self._endpoint.set_invoke()

    def set_tensor(self, tensor_index, value):
        self._verify_allocated()
        # self._endpoint.set_tensor(tensor_index, value.tobytes())

    def get_tensor(self, tensor_index):
        self._verify_allocated()

        tensor_details = self.get_tensor_details()[tensor_index]

        buffer = self._endpoint.get_tensor(tensor_index)
        # self._endpoint.clear()

        tensor = np.frombuffer(buffer, tensor_details["dtype"])
        tensor = tensor.reshape(tensor_details["shape"])

        return tensor
