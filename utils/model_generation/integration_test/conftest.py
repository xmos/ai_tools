# Copyright (c) 2020, XMOS Ltd, All rights reserved

import os
import pytest
import logging

import numpy as np

from xcore_model_generation.utils import stringify_config
from xcore_model_generation.model_generator import IntegrationTestModelGenerator


#  ----------------------------------------------------------------------------
#                                   HOOKS
#  ----------------------------------------------------------------------------


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
            ids=[f"CONFIGS[{j}]" for j, _ in enumerate(configs)],
        )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture
def run(request):
    try:
        GENERATOR = request.module.GENERATOR
    except AttributeError:
        raise NameError("GENERATOR not designated in test") from None

    gen = GENERATOR()
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
        logging.debug(f"cached generator loaded from {dirpath}")
    else:
        dirpath = os.path.join(pytest_config.cache.makedir("model_cache"), config_str)
        gen.run()
        gen.save(dirpath, dump_models=pytest_config.getoption("dump") == "models")
        logging.debug(f"generator cached to {dirpath}")
        pytest_config.cache.set(key, dirpath)

    if pytest_config.getoption("--generate-only"):
        pytest.skip()

    return gen.run


#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------


def _test_batched_arrays(predicted, expected, tolerance=1):
    failures = []
    assert predicted.shape[0] == expected.shape[0]
    for j, (arr, arr_ref) in enumerate(zip(predicted, expected)):
        diff = np.abs(np.int32(arr) - np.int32(arr_ref))
        diff_idx = zip(*np.where(diff > tolerance))

        failures.extend(
            f"Example {j}, idx={idx}: diff={diff[idx]}, "
            f"expected={arr_ref[idx]}, predicted={arr[idx]}"
            for idx in diff_idx
        )
    return failures


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_output(run, request):
    failures = _test_batched_arrays(run.outputs.xcore, run.outputs.reference)
    if failures:
        pytest.fail(
            f"\n{request.node.fspath}::{request.node.name}\n" + "\n".join(failures),
            pytrace=False,
        )
