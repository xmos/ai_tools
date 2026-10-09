import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from xmos_ai_tools import xformer
from xmos_ai_tools.xinterpreters import TFLMHostInterpreter

cwd = Path(__file__).resolve().parent

THREAD_COUNT = 1
INT8_RTOL = 0
INT8_ATOL = 1
FLOAT_RTOL = 1e-5
FLOAT_ATOL = 1e-6
RANDOM_SEED = 42
MODEL_FOLDER = cwd.parent / "models"

compile_options = {
    "XK-EVK-XU316": [("xcore-thread-count", THREAD_COUNT)],
    "XK-EVK-XU416": [
        ("xcore-thread-count", THREAD_COUNT),
        ("xcore-target-arch", "VX4A"),
    ],
}

compiler_flags = {
    "XK-EVK-XU316": "-O3;-g;-mcmodel=large",
    "XK-EVK-XU416": "-O3;-g",
}


def get_model_files():
    with (cwd / "models.yaml").open() as model_list:
        patterns = yaml.safe_load(model_list)
    models = []
    for pattern in patterns:
        matches = sorted(MODEL_FOLDER.rglob(pattern))
        models.extend(matches)
    return models


def compile_custom_model(model_path, cfg_name, hw_target="XK-EVK-XU316"):
    model_source = os.path.relpath(Path(model_path).resolve(), cwd)
    build_dir = cwd / "build" / cfg_name
    includes_dir = str(Path(model_path).resolve().parent)
    flags = compiler_flags.get(hw_target, compiler_flags["XK-EVK-XU316"])
    subprocess.run(
        [
            "cmake",
            "-B",
            f"{build_dir}",
            f"-DAPP_HW_TARGET={hw_target}",
            f"-DAPP_CXX_SRCS=src/main.cpp;{model_source}",
            f"-DAPP_INCLUDES={includes_dir}",
            f"-DAPP_COMPILER_FLAGS_{cfg_name}={flags}",
        ],
        cwd=cwd,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", cfg_name],
        cwd=cwd,
        check=True,
    )


def generate_input_random(shape, dtype, seed=RANDOM_SEED):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.bool_):
        return rng.integers(0, 2, size=shape, dtype=np.uint8).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        return rng.random(shape).astype(dtype)
    limits = np.iinfo(dtype)
    return rng.integers(limits.min, limits.max, size=shape, dtype=dtype, endpoint=True)


def get_tolerances(dtype):
    if np.issubdtype(dtype, np.floating):
        return FLOAT_RTOL, FLOAT_ATOL
    return INT8_RTOL, INT8_ATOL


def run_custom_model_host(model_path, input_name, output_name):
    interpreter = TFLMHostInterpreter()
    try:
        interpreter.set_model(model_path=str(model_path))
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        inputs = generate_input_random(input_details["shape"], input_details["dtype"])
        inputs.tofile(input_name)
        interpreter.set_tensor(0, inputs)
        interpreter.invoke()
        outputs = interpreter.get_tensor(output_details["index"]).copy()
        outputs.tofile(output_name)
        return inputs, outputs
    finally:
        interpreter.close()


def run_custom_model_device(cfg_name, work_dir):
    binary_path = cwd / "bin" / cfg_name / f"app_no_flash_{cfg_name}.xe"
    subprocess.run(
        ["xsim", str(binary_path)],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )


def pipeline_model(model_path: Path, hw_target="XK-EVK-XU316"):
    with tempfile.TemporaryDirectory(prefix="device_sim_") as temp_dir:
        work_dir = Path(temp_dir)
        cfg_name = work_dir.name
        exported_model = work_dir / "model.tflite"
        exported_model_cpp = str(exported_model) + ".cpp"
        input_file = work_dir / "input.bin"
        host_file = work_dir / "host_output.bin"
        sim_file = work_dir / "sim_output.bin"

        compiler_options = compile_options.get(
            hw_target, compile_options["XK-EVK-XU316"]
        )
        xformer.convert(model_path, exported_model, compiler_options)
        inputs, host_outputs = run_custom_model_host(
            exported_model, input_file, host_file
        )
        compile_custom_model(exported_model_cpp, cfg_name, hw_target)
        run_custom_model_device(cfg_name, work_dir)
        sim_outputs = np.fromfile(sim_file, dtype=host_outputs.dtype).reshape(
            host_outputs.shape
        )
        return host_outputs, sim_outputs


@pytest.mark.parametrize(
    "model_file",
    get_model_files(),
    ids=lambda model: str(model.relative_to(MODEL_FOLDER)),
)
def test_pipeline(model_file, request):
    hw_target = request.config.getoption("hw_target")
    request.node.user_properties.append(
        ("model", str(model_file.relative_to(MODEL_FOLDER)))
    )
    request.node.user_properties.append(("hw_target", hw_target))
    host_outputs, sim_outputs = pipeline_model(model_file, hw_target)
    rtol, atol = get_tolerances(host_outputs.dtype)
    host_values = host_outputs.astype(np.float64)
    sim_values = sim_outputs.astype(np.float64)
    differences = np.abs(sim_values - host_values)
    matches = np.isclose(sim_values, host_values, rtol=rtol, atol=atol, equal_nan=True)
    request.node.user_properties.extend(
        [
            ("max_abs_diff", float(differences.max(initial=0))),
            ("mean_abs_diff", float(differences.mean()) if differences.size else 0),
            ("values_outside_tolerance", int(np.count_nonzero(~matches))),
            ("total_values", int(differences.size)),
            ("rtol", rtol),
            ("atol", atol),
        ]
    )
    np.testing.assert_allclose(sim_values, host_values, rtol=rtol, atol=atol)


if __name__ == "__main__":
    model_path = MODEL_FOLDER / "8x8/test_add/test_add_0.tflite"
    host_outputs, sim_outputs = pipeline_model(model_path)
    rtol, atol = get_tolerances(host_outputs.dtype)
    np.testing.assert_allclose(sim_outputs, host_outputs, rtol=rtol, atol=atol)
    print(f"PASS: {model_path.name}")
