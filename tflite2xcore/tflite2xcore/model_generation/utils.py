# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf
from collections import Iterable
from typing import Union, Any, Tuple

from . import Configuration


class RandomUniform(tf.keras.initializers.RandomUniform):  # type: ignore
    def __call__(
        self, shape: Tuple[int, ...], dtype: tf.dtypes.DType = None
    ) -> tf.Tensor:
        try:
            return super().__call__(shape, dtype)
        except Exception as e:
            if e.args[0].startswith("Invalid dtype "):
                dtype = tf.dtypes.as_dtype(dtype)
                if dtype in (tf.int8, tf.int16):
                    if self.minval < dtype.min:
                        raise ValueError(
                            f"initializer minval = {self.minval} < {dtype.min} = dtype.min"
                        ) from None
                    elif self.maxval > dtype.max:
                        raise ValueError(
                            f"initializer maxval = {self.maxval} < {dtype.max} = dtype.max"
                        ) from None
                    else:
                        return tf.cast(
                            self._random_generator.random_uniform(
                                shape, self.minval, self.maxval, tf.int32
                            ),
                            dtype,
                        )
            raise


def parse_init_config(
    name: str, *args: Union[int, float]
) -> tf.keras.initializers.Initializer:
    if name == "RandomUniform":
        init = RandomUniform
    else:
        init = getattr(tf.keras.initializers, name)
    return init(*args)


def stringify_config(cfg: Configuration) -> str:
    def stringify_value(v: Any) -> str:
        if not isinstance(v, str) and isinstance(v, Iterable):
            v = "(" + ",".join(str(c) for c in v) + ")"
        return str(v).replace(" ", "_")

    return ",".join(k + "=" + stringify_value(v) for k, v in sorted(cfg.items()))
