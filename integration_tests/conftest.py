import glob
import pathlib
import sys

import pytest

# workaround to get debug logs when using xdist
sys.stdout = sys.stderr


def pytest_addoption(parser):
    parser.addoption(
        "--s",
        default=False,
        action="store_true",
        help="sleep for five seconds to attach a debugger",
    )
    parser.addoption(
        "--bnn", default=False, action="store_true", help="run binarized models"
    )
    parser.addoption(
        "--device", default=False, action="store_true", help="run tests on xcore"
    )
    parser.addoption(
        "--compiled", default=False, action="store_true", help="test compiled models"
    )
    parser.addoption(
        "--models_path",
        action="store",
        type=pathlib.Path,
        help="path to the directory containing the models to be tested",
    )
    parser.addoption(
        "--skip-version-check", action="store_true", help="skip the lib_nn version check"
    )
    parser.addoption(
        "--tc",
        action="store",
        default=5,
        type=int,
        help="xcore-thread-count parameter for compilation",
    )


def pytest_generate_tests(metafunc):
    if "filename" not in metafunc.fixturenames:
        return
    models_path = metafunc.config.getoption("models_path")
    if models_path is None:
        raise pytest.UsageError("--models_path is required for model tests")
    filelist = glob.glob(str(models_path) + "/**/*.tflite", recursive=True)
    metafunc.parametrize("filename", filelist)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-version-check"):
        for item in items:
            if item.path.name == "test_version_check.py":
                item.add_marker(pytest.mark.skip(reason="lib_nn version check disabled"))
