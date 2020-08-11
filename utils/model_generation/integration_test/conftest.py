# Copyright (c) 2020, XMOS Ltd, All rights reserved

import os
import pytest  # type: ignore
import _pytest  # type: ignore # NOTE: for typing only

from tflite2xcore import xlogging  # type: ignore # TODO: fix this

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
        coverage = metafunc.config.getoption("coverage")
        try:
            configs = metafunc.module.CONFIGS[coverage]
        except KeyError:
            raise KeyError(
                f"CONFIGS does not define coverage level '{coverage}'"
            ) from None
        except AttributeError:
            xlogging.logging.error(f"CONFIGS undefined in {metafunc.module}")
            configs = []

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
    key = "model_cache/" + config_str
    dirpath = pytest_config.cache.get(key, "")
    if dirpath:
        gen = IntegrationTestModelGenerator.load(dirpath)
        xlogging.logging.debug(f"cached generator loaded from {dirpath}")
    else:
        dirpath = os.path.join(pytest_config.cache.makedir("model_cache"), config_str)
        gen.run()
        gen.save(dirpath, dump_models=pytest_config.getoption("dump") == "models")
        xlogging.logging.debug(f"generator cached to {dirpath}")
        pytest_config.cache.set(key, dirpath)

    if pytest_config.getoption("--generate-only"):
        pytest.skip()

    return gen.run
