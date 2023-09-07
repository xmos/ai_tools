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
from abc import ABC, abstractmethod, abstractproperty

tf.keras.utils.set_random_seed(42)

MAX_ABS_ERROR = 1
ABS_AVG_ERROR = 1.0 / 4
AVG_ABS_ERROR = 1.0 / 4
REQUIRED_OUTPUTS = 2048
LOGGER = logging.getLogger(__name__)

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
LIB_TFLM_DIR_PATH = ROOT_DIR / "third_party" / "lib_tflite_micro"
LIB_NN_INCLUDE_PATH = ROOT_DIR / "third_party" / "lib_nn"
LIB_TFLM_INCLUDE_PATH = LIB_TFLM_DIR_PATH
TFLM_INCLUDE_PATH = pathlib.Path.joinpath(
    LIB_TFLM_DIR_PATH, "lib_tflite_micro", "submodules", "tflite-micro"
)
FLATBUFFERS_INCLUDE_PATH = pathlib.Path.joinpath(
    LIB_TFLM_DIR_PATH, "lib_tflite_micro", "submodules", "flatbuffers", "include"
)


class AbstractRunner(ABC):
    @abstractmethod
    def predict(self, inputs: list) -> list:
        pass


class RefRunner(AbstractRunner, ABC):
    @abstractproperty
    def input_details(self) -> tuple:
        """Abstract property to get input details of model

        Returns:
            tuple[list[type], list[list[int]]]
            A tuple containing a list of dtypes and a list of shapes
            expected by each of the model's inputs.
        """
        pass


class BnnInterpreter(RefRunner):
    def __init__(self, model_content):
        LOGGER.info("Creating LCE interpreter")
        self._interpreter = lce.testing.Interpreter(
            model_content, num_threads=1, use_reference_bconv=True
        )

    def predict(self, inputs):
        LOGGER.info("Invoking LCE interpreter")
        outs = self._interpreter.predict(inputs)
        return [outs] if len(outs) == 1 else outs

    @property
    def input_details(self):
        return self._interpreter.input_types, self._interpreter.input_shapes


class TFLiteInterpreter(RefRunner):
    def __init__(self, model_content):
        LOGGER.info("Creating TFLite interpreter")
        self._interpreter = tf.lite.Interpreter(
            model_content=model_content,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
            experimental_preserve_all_tensors=True,
        )
        self._interpreter.allocate_tensors()
        dets = self._interpreter.get_input_details()
        self._in_ids = [i["index"] for i in dets]
        self._in_shapes = [i["shape"] for i in dets]
        self._in_types = [i["dtype"] for i in dets]

    def predict(self, inputs):
        LOGGER.info("Invoking TFLite interpreter")
        self._interpreter.reset_all_variables()
        for idx, input in zip(self._in_ids, inputs):
            self._interpreter.set_tensor(idx, input)
        self._interpreter.invoke()
        dets = self._interpreter.get_output_details()
        return [self._interpreter.get_tensor(i["index"]) for i in dets]

    @property
    def input_details(self):
        return self._in_types, self._in_shapes


class XFHostInterpreter(AbstractRunner):
    def __init__(self, model_content):
        self._temp_dir = tempfile.TemporaryDirectory(suffix=str(os.getpid()))
        model = get_xformed_model(model_content, self._temp_dir.name)
        self._interpreter = xcore_tflm_host_interpreter()
        self._interpreter.set_model(model_content=model, secondary_memory=False)

    def predict(self, inputs):
        self._interpreter.reset()
        for i, j in enumerate(inputs):
            self._interpreter.set_tensor(i, j)
        self._interpreter.invoke()
        dets = self._interpreter.get_output_details()
        return [self._interpreter.get_tensor(i["index"]) for i in dets]

    def __del__(self):
        self._interpreter.close()
        self._temp_dir.cleanup()


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


def get_input_tensors(runner: AbstractRunner, parent_dir: pathlib.Path) -> list:
    types, shapes = runner.input_details
    ins = []
    for i, (s, d) in enumerate(zip(shapes, types)):
        f = parent_dir.joinpath(f"in{i+1}.npy")
        if f.is_file():
            ins.append(np.load(f))
        else:
            ins.append(np.random.randint(-128, high=127, size=s, dtype=d))
    return ins


# Run the model on Larq/TFLite interpreter,
# compare the output with xformed model on XCore TFLM
def test_model(request: FixtureRequest, filename: str) -> None:
    # for attaching a debugger
    opt_dict = {i: request.config.getoption(i) for i in ["bnn", "device", "s"]}
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

    if opt_dict["bnn"]:
        ref_interpreter = BnnInterpreter(model_content)
    else:
        ref_interpreter = TFLiteInterpreter(model_content)

    # some models are special (such as detection_postprocess), they don't have
    # a reference int8 TFLite operator and need to be loaded separately
    special_path = model_path.parent.joinpath("special.tflite.xc")
    if special_path.is_file():
        LOGGER.info("Processing special model")
        with open(special_path, "rb") as fd:
            model_content = fd.read()

    LOGGER.info("Invoking xformer to get xformed model...")
    xf_interpreter = XFHostInterpreter(model_content)

    # Run tests
    num_fails = run_out_count = run_out_err = run_out_abs_err = test = 0
    while run_out_count < params["REQUIRED_OUTPUTS"]:
        LOGGER.info(f"Run #{test}")
        test += 1
        input_tensors = get_input_tensors(ref_interpreter, model_path.parent)
        ref_outputs = ref_interpreter.predict(input_tensors)
        xf_outputs = xf_interpreter.predict(input_tensors)
        # Compare outputs
        errors = np.concatenate(
            [(a - b).reshape(-1) for a, b in zip(ref_outputs, xf_outputs)]
        )

        if len(errors) > 0:
            run_out_count += np.prod(errors.shape)
            run_out_err += np.sum(errors)
            run_out_abs_err += np.sum(np.abs(errors))
            max_abs_error = np.max(np.abs(errors))
            if max_abs_error > params["MAX_ABS_ERROR"]:
                LOGGER.error(f"Max abs error is too high: {max_abs_error}")
                assert max_abs_error <= 1

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

    assert num_fails == 0
