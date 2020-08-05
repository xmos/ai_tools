# Copyright (c) 2020, XMOS Ltd, All rights reserved

# TODO: move this file to final location

import pytest

import tensorflow as tf
import numpy as np

from conftest import Conv2dGenericTestModelGenerator
from xcore_model_generation.model_generator import Configuration


class Conv2dTestModelGenerator(Conv2dGenericTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        try:
            assert (
                cfg["K_h"] != 1 or cfg["K_w"] != 1
            ), "1x1 kernel is reserved for conv2d_1x1 testing"
        except KeyError:
            pass
        super()._set_config(cfg)


GENERATOR = Conv2dTestModelGenerator

CONFIGS = {
    "default": [
        {},
        {"height": 1, "width": 1},
        {"height": 3, "width": 3},
        {"height": 7, "width": 4, "input_channels": 32},
        {"height": 8, "width": 4, "output_channels": 16},
        {
            "height": 6,
            "width": 9,
            "input_channels": 24,
            "output_channels": 20,
            "padding": "valid",
        },
        {"height": 3, "width": 3, "padding": "valid"},
        {"strides": (2, 2)},
        {"strides": (1, 2)},
        {"strides": (1, 2), "padding": "valid"},
        {"strides": (2, 2), "padding": "valid"},
        {"height": 10, "width": 20, "num_threads": 2},
        {"height": 20, "width": 10, "num_threads": 5},
        {"height": 20, "width": 10, "num_threads": 5, "padding": "valid"},
        {"K_h": 2, "K_w": 4},
        {"K_h": 1, "K_w": 5, "padding": "valid"},
        {"K_h": 3, "K_w": 1, "padding": "valid"},
        {"K_h": 2, "K_w": 4, "input_channels": 20, "output_channels": 20},
        {
            "height": 7,
            "width": 8,
            "K_h": 2,
            "K_w": 4,
            "input_channels": 28,
            "output_channels": 12,
            "strides": (2, 1),
        },
    ],
}


def test_foo(run):
    for arr, arr_ref in zip(run.outputs.xcore, run.outputs.reference):
        assert np.max(np.abs(np.int32(arr) - np.int32(arr_ref))) <= 1


if __name__ == "__main__":
    pytest.main()
