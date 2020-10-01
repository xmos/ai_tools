# Copyright (c) 2020, XMOS Ltd, All rights reserved

import os
import logging
import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore
from collections import Iterable
from typing import Union, Iterator, List, Optional, Any

from . import Configuration


def parse_init_config(
    name: str, *args: Union[int, float]
) -> tf.keras.initializers.Initializer:
    init = getattr(tf.keras.initializers, name)
    return init(*args)


def stringify_config(cfg: Configuration) -> str:
    def stringify_value(v: Any) -> str:
        if not isinstance(v, str) and isinstance(v, Iterable):
            v = "(" + ",".join(str(c) for c in v) + ")"
        return str(v).replace(" ", "_")

    return ",".join(k + "=" + stringify_value(v) for k, v in sorted(cfg.items()))
