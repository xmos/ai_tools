# Copyright (c) 2020, XMOS Ltd, All rights reserved

# TODO: move this file to final location

import pytest

import tensorflow as tf
import numpy as np

from conftest import Conv2dGenericTestModelGenerator
from xcore_model_generation.model_generator import Configuration


class Conv2d1x1TestModelGenerator(Conv2dGenericTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        assert cfg.setdefault("K_h", 1) == 1, "Kernel height must be 1"
        assert cfg.setdefault("K_w", 1) == 1, "Kernel width must be 1"
        super()._set_config(cfg)


GENERATOR = Conv2d1x1TestModelGenerator

CONFIGS = {
    "default": [
        {},
        {"height": 2, "width": 2},
        {"height": 3, "width": 6},
        {"input_channels": 8, "output_channels": 12},
        {"input_channels": 32, "output_channels": 16},
        {"height": 8, "width": 2, "input_channels": 12, "output_channels": 64},
        {
            "weight_init": tf.initializers.Constant(1.0),
            "bias_init": tf.initializers.Constant(1.0),
            "input_init": tf.initializers.Constant(1.0),
        },
        {
            "height": 1,
            "width": 4,
            "weight_init": tf.initializers.RandomNormal(0.0, 0.1),
            "bias_init": tf.initializers.RandomNormal(-0.5, 0.02),
            "input_init": tf.initializers.RandomUniform(10.0),
        },
        {"num_threads": 2},
        {"num_threads": 5},
    ],
}


def test_foo(run):
    for arr, arr_ref in zip(run.outputs.xcore, run.outputs.reference):
        assert np.max(np.abs(np.int32(arr) - np.int32(arr_ref))) <= 1


if __name__ == "__main__":
    pytest.main()
