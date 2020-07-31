# Copyright (c) 2020, XMOS Ltd, All rights reserved

# TODO: move this file to final location

import pytest
import logging

import tensorflow as tf

from xcore_model_generation.model_generator import (
    IntegrationTestModelGenerator,
    Configuration,
)


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


class Conv2dTestModelGenerator(IntegrationTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        input_channels = cfg.pop("input_channels", 32)
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        output_channels = cfg.pop("output_channels", 16)
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
            weight_init=cfg.pop(
                "weight_init", tf.random_uniform_initializer(-1, 1)
            ),  # TODO: fix init config parameters
            bias_init=cfg.pop("bias_init", tf.constant_initializer(0)),
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
                    bias_initializer=cfg["bias_init"],
                    kernel_initializer=cfg["weight_init"],
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

