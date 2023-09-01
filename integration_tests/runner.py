import time
import pathlib
import logging
import tempfile
from _pytest.fixtures import FixtureRequest
import numpy as np
import os
import sys
import subprocess
import larq_compute_engine as lce
import tensorflow as tf
from xmos_ai_tools.xinterpreters import xcore_tflm_host_interpreter
from xmos_ai_tools import xformer
import yaml

np.random.seed(0)

MAX_ABS_ERROR = 1
ABS_AVG_ERROR = 1.0 / 4
AVG_ABS_ERROR = 1.0 / 4
REQUIRED_OUTPUTS = 2048
LOGGER = logging.getLogger(__name__)

LIB_TFLM_DIR_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "third_party" / "lib_tflite_micro"
)
LIB_NN_INCLUDE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "third_party" / "lib_nn"
)
LIB_TFLM_INCLUDE_PATH = LIB_TFLM_DIR_PATH
TFLM_INCLUDE_PATH = pathlib.Path.joinpath(
    LIB_TFLM_DIR_PATH, "lib_tflite_micro", "submodules", "tflite-micro"
)
FLATBUFFERS_INCLUDE_PATH = pathlib.Path.joinpath(
    LIB_TFLM_DIR_PATH, "lib_tflite_micro", "submodules", "flatbuffers", "include"
)


def run_cmd(cmd, working_dir=None):
    try:
        subprocess.run(
            cmd,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(e.output)
        sys.exit(1)


def get_xformed_model(model: bytes, temp_dirname) -> bytes:
    # write input model to temporary file
    input_file = pathlib.Path(temp_dirname) / "input.tflite"
    print(input_file)
    with open(input_file, "wb") as fd:
        fd.write(model)
    # create another temp file for output model
    output_file = pathlib.Path(temp_dirname) / "model.tflite"
    hyper_params = {"xcore-thread-count": 5}
    xformer.convert(input_file, output_file, hyper_params)

    # read output model content
    with open(output_file, "rb") as fd:
        bits = fd.read()
    return bits


def get_params(model_path: pathlib.Path) -> dict:
    params = {}
    params["MAX_ABS_ERROR"] = MAX_ABS_ERROR
    params["ABS_AVG_ERROR"] = ABS_AVG_ERROR
    params["AVG_ABS_ERROR"] = AVG_ABS_ERROR
    params["REQUIRED_OUTPUTS"] = REQUIRED_OUTPUTS
    yaml_filename = os.path.join(os.path.dirname(model_path), "params.yaml")
    if os.path.exists(yaml_filename):
        with open(yaml_filename) as f:
            params.update(yaml.safe_load(f))
    return params


def parse_request(request: FixtureRequest) -> dict:
    opt_list = ["bnn", "device", "s"]
    opt_dict = {opt: request.config.getoption(opt) for opt in opt_list}
    opt_dict["detection_postprocess"] = "detection_postprocess" in request.node.name
    return opt_dict


class InterpreterAdapter:
    def __init__(self, interpreter):
        self._interpreter = interpreter
        if isinstance(interpreter, lce.testing.Interpreter):
            self._is_bnn = True
            self._num_inputs = len(interpreter.input_types)
            self._num_outputs = len(interpreter.output_types)
        elif isinstance(interpreter, tf.lite.Interpreter):
            interpreter.allocate_tensors()
            self._is_bnn = False
            self._num_inputs = len(interpreter.get_input_details())
            self._num_outputs = len(interpreter.get_output_details())
        else:
            raise AttributeError(f"Unknown interpreter: {interpreter.__class__}")
        self._input_tensors = None
        self._output_tensors = None

    def reset_variables(self) -> None:
        """Reset all stateful variable in interpreter (call after invoking RNN's)"""
        if self._is_bnn:
            # larq doesn't support that? We're not using binary LSTM's anyway
            pass
        else:
            self._interpreter.reset_all_variables()

    def set_tensors(self, tensors: list) -> None:
        assert len(tensors) == self._num_inputs
        if self._is_bnn:
            self._input_tensors = tensors
        else:
            for i in range(self._num_inputs):
                self._interpreter.set_tensor(
                    self._interpreter.get_input_details()[i]["index"], tensors[i]
                )

    def invoke(self) -> None:
        if self._is_bnn:
            assert self._input_tensors is not None
            self._output_tensors = self._interpreter.predict(self._input_tensors)
            if len(self._output_tensors) == 1:
                self._output_tensors = [self._output_tensors]
        else:
            self._interpreter.invoke()

    @property
    def outputs(self):
        if self._is_bnn:
            return self._output_tensors
        out_dets = self._interpreter.get_output_details()
        return [self._interpreter.get_tensor(det["index"]) for det in out_dets]

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def input_types(self):
        if self._is_bnn:
            return list(self._interpreter.input_types)
        else:
            return [
                self._interpreter.get_input_details()[i]["dtype"]
                for i in range(self._num_inputs)
            ]

    @property
    def input_shapes(self):
        if self._is_bnn:
            return list(self._interpreter.input_shapes)
        else:
            return [
                self._interpreter.get_input_details()[i]["shape"]
                for i in range(self._num_inputs)
            ]


def get_interpreter(model_content, opt_dict):
    if opt_dict["bnn"]:
        LOGGER.info("Creating LCE interpreter...")
        temp_interpreter = lce.testing.Interpreter(
            model_content, num_threads=1, use_reference_bconv=True
        )
    else:
        LOGGER.info("Creating TFLite interpreter...")
        temp_interpreter = tf.lite.Interpreter(
            model_content=model_content,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
            experimental_preserve_all_tensors=True,
        )
    return InterpreterAdapter(temp_interpreter)


def get_input_tensors(
    interpreter: InterpreterAdapter, parent_dir: pathlib.Path
) -> list:
    # hacky but should be enough
    if parent_dir.joinpath("in1.npy").is_file():
        LOGGER.info("Detected input files - loading...")
        num = interpreter.num_inputs
        return [np.load(parent_dir.joinpath(f"in{i+1}.npy")) for i in range(num)]
    LOGGER.info("Creating random input...")
    shapes, types = interpreter.input_shapes, interpreter.input_types
    return [
        np.random.randint(-128, high=127, size=s, dtype=dt)
        for (s, dt) in zip(shapes, types)
    ]


def get_interpreter_outputs(
    interpreter: InterpreterAdapter, input_tensors: list
) -> list:
    interpreter.reset_variables()
    interpreter.set_tensors(input_tensors)
    LOGGER.info("Invoking interpreter...")
    interpreter.invoke()
    return interpreter.outputs


# Run the model on Larq/TFLite interpreter and compare the output with xformed model on XCore TFLM
def test_model(request: FixtureRequest, filename: str) -> None:
    # for attaching a debugger
    opt_dict = parse_request(request)
    if opt_dict["s"]:
        time.sleep(5)

    model_path = pathlib.Path(filename).resolve()
    params = get_params(model_path)

    if not model_path.exists():
        LOGGER.error("Model file not found!")
        assert False
    # read in model from file
    with open(model_path, "rb") as fd:
        model_content = fd.read()

    interpreter = get_interpreter(model_content, opt_dict)

    LOGGER.info("Invoking xformer to get xformed model...")
    if opt_dict["detection_postprocess"]:
        LOGGER.info(
            "Detection postprocess special case - loading int8 model for xcore..."
        )
        with open(model_path.parent.joinpath("test_dtp.xc"), "rb") as fd:
            model_content = fd.read()

    temp_dirname = tempfile.TemporaryDirectory(suffix=str(os.getpid()))
    xformed_model = get_xformed_model(model_content, temp_dirname.name)

    LOGGER.info("Creating TFLM XCore interpreter...")
    ie = xcore_tflm_host_interpreter()
    ie.set_model(model_content=xformed_model, secondary_memory=False)

    # Run tests
    num_fails = run_out_count = run_out_err = run_out_abs_err = test = 0
    while run_out_count < params["REQUIRED_OUTPUTS"]:
        LOGGER.info(f"Run #{test}")
        test += 1
        input_tensors = get_input_tensors(interpreter, model_path.parent)
        outputs = get_interpreter_outputs(interpreter, input_tensors)
        num_outputs = len(outputs)
        ie.reset()

        LOGGER.info("Invoking XCORE interpreter...")
        for i in range(interpreter.num_inputs):
            ie.set_tensor(i, input_tensors[i])
        ie.invoke()
        xf_outputs = []
        for i in range(num_outputs):
            output_tensor = ie.get_tensor(ie.get_output_details()[i]["index"])
            LOGGER.info("outputs: " + str(output_tensor.shape))
            xf_outputs.append(output_tensor)

        # Compare outputs
        errors = np.concatenate(
            [(a - b).reshape(-1) for a, b in zip(outputs, xf_outputs)]
        )

        if len(errors) > 0:
            run_out_count += np.prod(errors.shape)
            run_out_err += np.sum(errors)
            run_out_abs_err += np.sum(np.abs(errors))
            max_abs_error = np.max(np.abs(errors))
            if max_abs_error > params["MAX_ABS_ERROR"]:
                LOGGER.error(f"Max abs error is too high: {max_abs_error}")
                assert max_abs_error <= 1

    np.set_printoptions(threshold=np.inf)
    avg_error = run_out_err / run_out_count
    avg_abs_error = run_out_abs_err / run_out_count
    LOGGER.info(f"{max_abs_error} {avg_error} {avg_abs_error}")

    def fail(msg: str) -> None:
        nonlocal num_fails
        num_fails += 1
        LOGGER.error(msg)
        LOGGER.error(f"Run #{test} failed")

    if abs(avg_error) > params["ABS_AVG_ERROR"]:
        fail(f"Abs avg error is too high: {abs(avg_error)}")

    if avg_abs_error > params["AVG_ABS_ERROR"]:
        fail(f"Avg abs error is too high: {avg_abs_error}")

    temp_dirname.cleanup()
    ie.close()
    assert num_fails == 0
