import glob
import pathlib
import sys

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
        required=True,
        help="path to the directory containing the models to be tested",
    )
    parser.addoption(
        "--tc",
        action="store",
        default=5,
        type=int,
        help="xcore-thread-count parameter for compilation",
    )


def pytest_generate_tests(metafunc):
    models_path = metafunc.config.getoption("models_path")
    filelist = glob.glob(str(models_path) + "/**/*.tflite", recursive=True)
    metafunc.parametrize("filename", filelist)
