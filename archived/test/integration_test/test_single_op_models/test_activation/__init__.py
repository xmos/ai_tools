# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import tensorflow as tf
from abc import abstractmethod
from typing import Callable, Optional, Tuple

from .. import (
    ChannelAgnosticOpTestModelGenerator,
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class LUTActivationOpTestModelGenerator(ChannelAgnosticOpTestModelGenerator):
    @property
    @abstractmethod
    def act_fun(self) -> Callable[[tf.Tensor], tf.Tensor]:
        raise NotImplementedError()

    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        kwargs = {"input_shape": input_shape} if input_shape else {}
        return tf.keras.layers.Lambda(self.act_fun, **kwargs)
