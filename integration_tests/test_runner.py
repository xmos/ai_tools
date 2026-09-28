"""Run the Jenkins daily host and device model tests locally."""

import platform
import subprocess
from pathlib import Path

import pytest

from runner import test_model as run_model


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "integration_tests" / "models"
HOST_TESTS = (
    ("float32", 1, False, False),
    ("16x8", 5, False, False),
    ("complex_models/8x8", 5, False, False),
    ("complex_models/float32", 5, False, False),
    ("8x8", 1, False, False),
    ("8x8", 5, False, False),
    ("bnns", 5, True, False),
    ("8x8", 5, False, True),
    ("bnns", 5, True, True),
    ("complex_models/8x8/test_mobilenet_v2", 5, False, True),
)
DEVICE_TESTS = (
    ("8x8/test_broadcast", 1),
    ("16x8/test_transpose", 5),
    ("8x8/test_concatenate", 5),
    ("8x8/test_mean", 1),
    ("16x8/test_mean", 1),
    ("8x8/test_lstm", 1),
    ("8x8/test_lstm", 5),
    ("complex_models/8x8/test_cnn_classifier", 1),
    ("complex_models/8x8/test_cnn_classifier", 5),
    ("8x8/test_softmax", 5),
    ("8x8/test_detection_postprocess", 5),
    ("16x8/test_conv2d", 5),
    ("16x8/test_transpose_conv", 5),
)


@pytest.mark.parametrize(
    "model_file, thread_count, bnn, compiled",
    [
        pytest.param(
            str(model),
            thread_count,
            bnn,
            compiled,
            id=f"{model.relative_to(MODELS_DIR)}-tc{thread_count}-bnn{bnn}-compiled{compiled}",
        )
        for models_path, thread_count, bnn, compiled in HOST_TESTS
        for model in sorted((MODELS_DIR / models_path).rglob("*.tflite"))
    ],
)
def test_daily_host(request, model_file, thread_count, bnn, compiled):
    if platform.system() == "Windows" and (bnn or compiled):
        pytest.skip("BNN and compiled host tests are disabled on Windows")
    run_model(request, model_file, thread_count=thread_count, bnn=bnn, compiled=compiled)


@pytest.fixture(scope="module")
def reset_device(request):
    if not request.config.getoption("--device"):
        pytest.skip("device tests require --device")
    last_group = None

    def reset_for_group(group):
        nonlocal last_group
        if group != last_group:
            subprocess.run(
                ["xtagctl", "reset_all", "XCORE-AI-EXPLORER"],
                cwd=ROOT_DIR,
                check=True,
            )
            last_group = group

    return reset_for_group


@pytest.mark.parametrize(
    "model_file, thread_count, model_group",
    [
        pytest.param(
            str(model),
            thread_count,
            (models_path, thread_count),
            id=f"{model.stem}-tc{thread_count}",
        )
        for models_path, thread_count in DEVICE_TESTS
        for model in sorted((MODELS_DIR / models_path).glob("*.tflite"))
    ],
)
def test_daily_device(request, reset_device, model_file, thread_count, model_group):
    reset_device(model_group)
    run_model(request, model_file, thread_count=thread_count)