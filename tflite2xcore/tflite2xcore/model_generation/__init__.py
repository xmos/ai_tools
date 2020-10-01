# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
from typing import Union, Dict, Any, TypeVar
from typing_extensions import Protocol

from tflite2xcore.utils import TFLiteModel  # type: ignore # TODO: fix this

Configuration = Dict[str, Any]

T_co = TypeVar("T_co", covariant=True)


class Hook(Protocol[T_co]):
    def __call__(self) -> T_co:
        ...


from .model_generators import ModelGenerator
