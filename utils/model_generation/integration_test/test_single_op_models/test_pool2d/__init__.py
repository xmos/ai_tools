# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
from abc import abstractmethod
from typing import Tuple, Optional

from tflite2xcore._model_generation import Configuration
from tflite2xcore._model_generation.utils import parse_init_config

from .. import (
    FilterOpTestModelGenerator,
    test_output,
    test_converted_single_op_model,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class Pool2dGenericTestModelGenerator(FilterOpTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        channels = cfg.pop("channels", 4)
        assert channels % 4 == 0, "# of channels must be multiple of 4"
        self._config.update({"channels": channels})

        strides = cfg.setdefault("strides", (2, 2))
        K_h = cfg.setdefault("K_h", 2)
        K_w = cfg.setdefault("K_w", 2)
        if cfg.setdefault("padding", "valid") == "same":
            assert (
                cfg["height"] % 2 == cfg["width"] % 2 == 0
                and K_h == K_w == 2
                and strides[0] == strides[1] == 2
            ), "same padding is only allowed for the common 2x2 case"

        super()._set_config(cfg)

    @property
    def _input_shape(self) -> Tuple[int, int, int]:
        cfg = self._config
        return cfg["height"], cfg["width"], cfg["channels"]
