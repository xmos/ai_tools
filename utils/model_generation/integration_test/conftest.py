# Copyright (c) 2020, XMOS Ltd, All rights reserved

import yaml
import logging
import portalocker  # type: ignore
import pytest  # type: ignore
import _pytest  # type: ignore # NOTE: for typing only
from pathlib import Path

from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore._model_generation.utils import stringify_config

from . import IntegrationTestModelGenerator, IntegrationTestRunner


#  ----------------------------------------------------------------------------
#                                   HOOKS
#  ----------------------------------------------------------------------------


def pytest_addoption(parser):  # type: ignore
    parser.addoption(
        "-C",
        "--coverage",
        action="store",
        default="default",
        choices=["reduced", "default", "extended"],
        help="Set the coverage level",
    )

    parser.addoption(
        "-D",
        "--dump",
        action="store",
        default=None,
        choices=[None, "models"],
        help="Set what contents of the model generation runs should be dumped into cache for easier access.",
    )

    parser.addoption(
        "--config-only",
        action="store_true",
        help="The model generators are configured but not run",
    )

    parser.addoption(
        "--generate-only",
        action="store_true",
        help="The model generators are run and cached but outputs are not evaluated for correctness",
    )


def pytest_generate_tests(metafunc: _pytest.python.Metafunc) -> None:
    if "run" in metafunc.fixturenames:
        try:
            CONFIGS = metafunc.module.CONFIGS  # [coverage].values()
            config_file = Path(metafunc.module.__file__)
        except AttributeError:
            logging.debug(f"CONFIGS undefined in {metafunc.module}")
            config_file = Path(metafunc.module.__file__).with_suffix(".yml")
            try:
                with open(config_file, "r") as f:
                    CONFIGS = yaml.load(f, Loader=yaml.FullLoader)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    "Cannot find .yml test config file and "
                    "test module does not contain CONFIGS"
                ) from e

        coverage = metafunc.config.getoption("coverage")
        try:
            configs = list(CONFIGS[coverage].values())
        except KeyError:
            raise KeyError(
                "CONFIGS does not define coverage level "
                f"'{coverage}' in {config_file.resolve()}"
            ) from None

        metafunc.parametrize(
            "run",
            configs,
            indirect=True,
            ids=[f"CONFIGS[{j}]" for j, _ in enumerate(configs)],
        )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def run(request: _pytest.fixtures.SubRequest) -> IntegrationTestRunner:
    try:
        GENERATOR = request.module.GENERATOR
    except AttributeError:
        raise NameError("GENERATOR not designated in test") from None

    gen: IntegrationTestModelGenerator = GENERATOR()
    gen.set_config(**request.param)

    pytest_config = request.config
    if pytest_config.getoption("verbose"):
        print(f"Config: {gen._config}")
    if pytest_config.getoption("--config-only"):
        pytest.skip()

    config_str = stringify_config(gen._config)
    file_path = Path(request.module.__file__)
    key = file_path.relative_to(pytest_config.rootdir) / config_str

    dirpath = pytest_config.cache.get(key, "")
    if dirpath:
        gen = IntegrationTestModelGenerator.load(dirpath)
        logging.debug(f"cached generator loaded from {dirpath}")
        gen.run.rerun_post_cache()
    else:
        gen.run()
        try:
            with portalocker.BoundedSemaphore(1, hash(key), timeout=0):
                dirpath = str(pytest_config.cache.makedir("model_cache") / key)
                gen.save(
                    dirpath, dump_models=pytest_config.getoption("dump") == "models",
                )
                logging.debug(f"generator cached to {dirpath}")
                pytest_config.cache.set(key, dirpath)
        except portalocker.AlreadyLocked:
            # another process will write to cache
            pass

    if pytest_config.getoption("--generate-only"):
        pytest.skip()

    return gen.run


@pytest.fixture  # type: ignore
def xcore_model(run: IntegrationTestRunner) -> XCOREModel:
    return XCOREModel.deserialize(run.models.xcore)


@pytest.fixture  # type: ignore
def xcore_identical_model(run: IntegrationTestRunner) -> XCOREModel:
    return XCOREModel.deserialize(run.models.xcore_identical)
