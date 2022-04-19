import glob
import pathlib


def pytest_addoption(parser):
    parser.addoption(
        "--bnn", default=False, action="store_true", help="run binarized models"
    )
    parser.addoption(
        "--number_of_samples",
        default=100,
        action="store",
        type=int,
        help="number of samples to run",
    )
    parser.addoption(
        "--models_path",
        action="store",
        type=pathlib.Path,
        required=True,
        help="path to the directory containing the models to be tested",
    )


def pytest_generate_tests(metafunc):
    models_path = metafunc.config.getoption("models_path")
    filelist = glob.glob(str(models_path) + "/**/*.tflite", recursive=True)
    metafunc.parametrize("filename", filelist)
    # if "stringinput" in metafunc.fixturenames:
    #     metafunc.parametrize("stringinput", metafunc.config.getoption("stringinput"))
