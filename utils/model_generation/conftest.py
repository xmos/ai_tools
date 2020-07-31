# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest
import logging


def pytest_addoption(parser):
    parser.addoption(
        "-C",
        "--coverage",
        action="store",
        default="default",
        choices=["reduced", "default", "extended"],
        help="Set the coverage level",
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


def pytest_generate_tests(metafunc):
    if "run" in metafunc.fixturenames:
        coverage = metafunc.config.getoption("coverage")
        try:
            configs = metafunc.module.CONFIGS[coverage]
        except KeyError:
            raise KeyError(
                f"CONFIGS does not define coverage level '{coverage}'"
            ) from None
        except AttributeError:
            logging.error(f"CONFIGS undefined in {metafunc.module}")
            configs = []

        metafunc.parametrize(
            "run",
            configs,
            indirect=True,
            ids=[f'CONFIGS["default"][{j}]' for j, _ in enumerate(configs)],
        )


@pytest.fixture
def run(request):
    try:
        GENERATOR = request.module.GENERATOR
    except AttributeError:
        raise NameError("GENERATOR not designated in test") from None

    generator = GENERATOR()
    generator.set_config(**request.param)
    logging.info(f"Full config: {generator._config}")
    if request.config.getoption("--config-only"):
        pytest.skip()

    # TODO: cache here
    generator.run()
    if request.config.getoption("--generate-only"):
        pytest.skip()

    return generator.run
