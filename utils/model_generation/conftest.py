# Copyright (c) 2020, XMOS Ltd, All rights reserved

# TODO: move this file to final location

import os
import pytest
import logging

import tensorflow as tf
import numpy as np

from xcore_model_generation.utils import parse_init_config, stringify_config
from xcore_model_generation.model_generator import (
    IntegrationTestModelGenerator,
    Configuration,
)


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
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Conv2dGenericTestModelGenerator(IntegrationTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        input_channels = cfg.pop("input_channels", 4)
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        output_channels = cfg.pop("output_channels", 4)
        assert output_channels % 4 == 0, "# of output channels must be multiple of 4"

        self._config = dict(
            K_w=cfg.pop("K_w", 3),
            K_h=cfg.pop("K_h", 3),
            height=cfg.pop("height", 5),
            width=cfg.pop("width", 5),
            input_channels=input_channels,
            output_channels=output_channels,
            padding=cfg.pop("padding", "same"),
            strides=cfg.pop("strides", (1, 1)),
            weight_init=cfg.pop("weight_init", ("RandomUniform", -1, 1)),
            bias_init=cfg.pop("bias_init", ("Constant", 0)),
        )
        super()._set_config(cfg)

    def build_core_model(self):
        cfg = self._config
        return tf.keras.Sequential(
            layers=[
                tf.keras.layers.Conv2D(
                    filters=cfg["output_channels"],
                    kernel_size=(cfg["K_h"], cfg["K_w"]),
                    padding=cfg["padding"],
                    strides=cfg["strides"],
                    input_shape=(cfg["height"], cfg["width"], cfg["input_channels"]),
                    bias_initializer=parse_init_config(*cfg["bias_init"]),
                    kernel_initializer=parse_init_config(*cfg["weight_init"]),
                )
            ],
        )

    def build(self):
        self._prep_backend()
        try:
            self._model = self.build_core_model()
        except ValueError as e:
            if e.args[0].startswith("Negative dimension size caused by"):
                raise ValueError(
                    "Negative dimension size (Hint: if using 'valid' padding "
                    "verify that the kernel is at least the size of input image)"
                ) from e
            else:
                raise
        self._model.build()


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
