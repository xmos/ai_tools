# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
from typing import Tuple, Optional

from . import Configuration
from .utils import parse_init_config
from .runners import RunnerDependent


class DataFactory(RunnerDependent):
    def _set_config(self, cfg: Configuration) -> None:
        if "input_init" not in self._config:
            self._config["input_init"] = cfg.pop("input_init", ("RandomUniform", -1, 1))
        super()._set_config(cfg)

    def make_data(
        self, shape: Tuple[int, ...], batch: Optional[int] = None
    ) -> tf.Tensor:
        if batch is not None:
            shape = (batch, *shape)

        init = parse_init_config(*self._config["input_init"])
        return init(shape)
